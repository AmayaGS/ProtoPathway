"""
Model Profiling Script for ProtoPathway.

Measures computational efficiency metrics across all models:
- Parameter count (trainable + total)
- FLOPs (PyTorch built-in + manual GNN estimation)
- Peak GPU memory (VRAM) during forward + backward
- Training time per patient (forward + backward + optimizer step)
- Inference time per patient (forward only)

Disentangles data loading from compute by pre-loading patients into memory.

Usage:
    python main.py profile --config configs/experiments/experiment.yaml
    python main.py profile --config configs/experiments/experiment.yaml --models protopath abmil snn
    python main.py profile --config configs/experiments/experiment.yaml --num-patients 50 --num-warmup 5
"""

import os
import json
import time
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader as PyGDataLoader
from omegaconf import OmegaConf

from utils.dataset import load_dataset_components, sample_wsi_embeddings, compute_centroids
from utils.losses import NLLSurvLoss
from models.factory import build_model, get_model_requirements, get_available_models


# =============================================================================
# FLOPs Estimation
# =============================================================================

def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def estimate_gnn_flops(model, data, cfg):
    """
    Manually estimate FLOPs for GNN operations that PyTorch's FlopCounterMode misses.

    SAGEConv: mean aggregation + linear transform per edge/node
    GATv2Conv: attention score computation + aggregation

    Returns estimated FLOPs as int.
    """
    flops = 0
    model_name = cfg.model.name.lower()
    hidden_dim = cfg.model.gene_encoder.hidden_dim

    # Check if model uses GNN (gene encoder)
    reqs = get_model_requirements(model_name)
    if not reqs.get('needs_graph', False):
        return 0

    # Get graph dimensions from data
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    input_dim = data.x.shape[1]

    if model_name == 'protopath':
        num_layers = cfg.model.gene_encoder.num_layers
        num_heads = cfg.model.gene_encoder.num_heads

        # SAGEConv layer 1: linear(input_dim -> hidden_dim) + mean aggregation
        # Linear: 2 * num_nodes * input_dim * hidden_dim (weight multiply + add)
        # Aggregation: num_edges * hidden_dim (scatter_mean)
        flops += 2 * num_nodes * input_dim * hidden_dim  # linear transform
        flops += num_edges * hidden_dim  # message passing aggregation

        # Middle SAGEConv layers
        for _ in range(num_layers - 2):
            flops += 2 * num_nodes * hidden_dim * hidden_dim
            flops += num_edges * hidden_dim

        # Final GATv2Conv layer
        # Per head: linear projections for Q,K,V, attention score, softmax, aggregation
        # Q/K projection: num_nodes * hidden_dim * hidden_dim
        # Attention score per edge: hidden_dim (LeakyReLU(a^T [Wq||Wk]))
        # Softmax: ~5 * num_edges (exp, sum, div)
        # Aggregation: num_edges * hidden_dim
        per_head = (
            2 * num_nodes * hidden_dim * (hidden_dim // num_heads)  # Q, K projections
            + num_edges * (hidden_dim // num_heads)  # attention scores
            + 5 * num_edges  # softmax
            + num_edges * (hidden_dim // num_heads)  # weighted aggregation
        )
        flops += per_head * num_heads

        # Gate: num_pathways * hidden_dim + softmax
        num_pathways = data.num_pathways
        flops += num_pathways * hidden_dim + 5 * num_pathways

    elif model_name in ('snn', 'survpath', 'mcat', 'motcat', 'pibd', 'mmp'):
        # These models use the pathway structure but via different mechanisms
        # SNN uses pathway_gene_indices to group genes, then MLP per pathway
        # For SNN: pathway grouping is essentially indexing + MLP forward
        # We'll estimate conservatively
        num_pathways = data.num_pathways if hasattr(data, 'num_pathways') else 50
        num_genes = data.num_genes if hasattr(data, 'num_genes') else 300

        if model_name == 'snn':
            # SNN processes genes per pathway through shared layers
            # Approximate: each pathway's genes go through hidden_dims MLP
            hidden_dims = cfg.model.gene_encoder.get('hidden_dims', [256, 256])
            avg_genes_per_pathway = num_genes * 2 // num_pathways  # rough avg (edges are bidirectional)
            for i, hd in enumerate(hidden_dims):
                in_d = avg_genes_per_pathway if i == 0 else hidden_dims[i - 1]
                flops += num_pathways * 2 * in_d * hd

    return flops


def estimate_flops_builtin(model, sample_batch, cfg, device):
    """
    Use PyTorch's built-in FlopCounterMode to count standard ops.

    Returns counted FLOPs (excludes GNN ops).
    """
    model.eval()
    total_flops = 0

    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as flop_counter:
            with torch.no_grad():
                if cfg.model.name == 'pibd':
                    _ = model(sample_batch)
                else:
                    _ = model(sample_batch, return_attention=False)
            total_flops = int(flop_counter.get_total_flops())
    except ImportError:
        logging.warning("FlopCounterMode not available (requires PyTorch >= 2.1). "
                        "Only manual GNN estimates will be reported.")
    except Exception as e:
        logging.warning(f"FlopCounterMode failed: {e}. Using manual estimates only.")

    return total_flops


# =============================================================================
# Memory Profiling
# =============================================================================

def measure_peak_memory(model, sample_batch, criterion, cfg, device, mode='train'):
    """
    Measure peak GPU memory during forward (+ backward for training).

    Args:
        mode: 'train' for forward+backward, 'inference' for forward only

    Returns:
        peak_memory_mb: Peak allocated memory in MB
    """
    if not torch.cuda.is_available():
        return 0.0

    model = model.to(device)
    sample_batch = sample_batch.to(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Baseline memory (model weights already on GPU)
    baseline = torch.cuda.memory_allocated(device)

    if mode == 'train':
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        if cfg.model.name == 'pibd':
            logits, aux_losses = model(sample_batch)
        else:
            logits = model(sample_batch, return_attention=False)

        if cfg.task == 'survival':
            target = sample_batch.y['bin']
            t = sample_batch.y['time']
            event = sample_batch.y['event']
            loss = criterion(logits, target, t, event)
        else:
            loss = criterion(logits, sample_batch.y)

        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            if cfg.model.name == 'pibd':
                _ = model(sample_batch)
            else:
                _ = model(sample_batch, return_attention=False)

    peak = torch.cuda.max_memory_allocated(device)
    peak_mb = peak / (1024 ** 2)

    # Clean up
    torch.cuda.empty_cache()

    return peak_mb


# =============================================================================
# Timing
# =============================================================================

def time_training(model, batches, criterion, cfg, device, num_warmup=5):
    """
    Time training iterations (forward + backward + optimizer step).
    Data is already pre-loaded in memory — no disk I/O in the loop.

    Returns:
        mean_ms: Mean time per patient in milliseconds
        std_ms: Std time per patient in milliseconds
    """
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Move all batches to device upfront
    batches_gpu = [b.to(device) for b in batches]

    # Warmup (not timed)
    for i in range(min(num_warmup, len(batches_gpu))):
        b = batches_gpu[i]
        optimizer.zero_grad()
        if cfg.model.name == 'pibd':
            logits, aux = model(b)
            if cfg.task == 'survival':
                loss = criterion(logits, b.y['bin'], b.y['time'], b.y['event'])
            else:
                loss = criterion(logits, b.y)
        else:
            logits = model(b, return_attention=False)
            if cfg.task == 'survival':
                loss = criterion(logits, b.y['bin'], b.y['time'], b.y['event'])
            else:
                loss = criterion(logits, b.y)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed iterations
    times = []
    for b in batches_gpu:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        if cfg.model.name == 'pibd':
            logits, aux = model(b)
            if cfg.task == 'survival':
                loss = criterion(logits, b.y['bin'], b.y['time'], b.y['event'])
            else:
                loss = criterion(logits, b.y)
        else:
            logits = model(b, return_attention=False)
            if cfg.task == 'survival':
                loss = criterion(logits, b.y['bin'], b.y['time'], b.y['event'])
            else:
                loss = criterion(logits, b.y)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return np.mean(times), np.std(times)


def time_inference(model, batches, cfg, device, num_warmup=5):
    """
    Time inference (forward only, no grad).
    Data is already pre-loaded in memory.

    Returns:
        mean_ms, std_ms: Time per patient in milliseconds
    """
    model = model.to(device)
    model.eval()

    batches_gpu = [b.to(device) for b in batches]

    # Warmup
    with torch.no_grad():
        for i in range(min(num_warmup, len(batches_gpu))):
            b = batches_gpu[i]
            if cfg.model.name == 'pibd':
                _ = model(b)
            else:
                _ = model(b, return_attention=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed
    times = []
    with torch.no_grad():
        for b in batches_gpu:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if cfg.model.name == 'pibd':
                _ = model(b)
            else:
                _ = model(b, return_attention=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return np.mean(times), np.std(times)


# =============================================================================
# Data Pre-loading (disentangles I/O from compute)
# =============================================================================

def preload_batches(dataset, num_patients, device='cpu'):
    n = min(num_patients, len(dataset))
    logging.info(f"  Pre-loading {n} patients into memory...")

    subset = torch.utils.data.Subset(dataset, range(n))
    loader = PyGDataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    t0 = time.perf_counter()
    batches = [batch for batch in loader]
    load_time = time.perf_counter() - t0

    logging.info(f"  Pre-loaded {n} patients in {load_time:.1f}s "
                 f"({load_time/n*1000:.1f} ms/patient)")

    return batches, load_time


# =============================================================================
# LaTeX Output
# =============================================================================

def generate_latex_table(results, output_path):
    """Generate a LaTeX table from profiling results."""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Computational efficiency comparison across models. "
                 r"Params = trainable parameters. "
                 r"FLOPs estimated per patient (single forward pass). "
                 r"VRAM = peak GPU memory during training. "
                 r"Train/Infer = time per patient excluding data loading.}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Modality & Params & FLOPs & VRAM (MB) & Train (ms) & Infer (ms) \\")
    lines.append(r"\midrule")

    # Sort: gene-only, wsi-only, multimodal
    modality_order = {'gene': 0, 'wsi': 1, 'multimodal': 2}
    sorted_results = sorted(results, key=lambda r: (modality_order.get(r['modality'], 3), r['model']))

    prev_modality = None
    for r in sorted_results:
        if prev_modality is not None and r['modality'] != prev_modality:
            lines.append(r"\midrule")
        prev_modality = r['modality']

        # Format params
        params = r['trainable_params']
        if params >= 1e6:
            params_str = f"{params/1e6:.2f}M"
        elif params >= 1e3:
            params_str = f"{params/1e3:.1f}K"
        else:
            params_str = str(params)

        # Format FLOPs
        flops = int(r['total_flops'])
        if flops >= 1e9:
            flops_str = f"{flops/1e9:.2f}G"
        elif flops >= 1e6:
            flops_str = f"{flops/1e6:.1f}M"
        elif flops >= 1e3:
            flops_str = f"{flops/1e3:.1f}K"
        else:
            flops_str = str(flops)

        # Format VRAM
        vram_str = f"{r['peak_vram_train_mb']:.0f}" if r['peak_vram_train_mb'] > 0 else "CPU"

        # Format times
        train_str = f"{r['train_ms_mean']:.1f}"
        infer_str = f"{r['infer_ms_mean']:.1f}"

        # Modality tag
        mod_map = {'gene': 'Gene', 'wsi': 'WSI', 'multimodal': 'Multi'}
        mod_str = mod_map.get(r['modality'], r['modality'])

        # Model display name
        name = r['model']
        display_names = {
            'protopath': r'\textbf{ProtoPathway}',
            'abmil': 'ABMIL',
            'transmil': 'TransMIL',
            'dsmil': 'DSMIL',
            'snn': 'SNN',
            'mlp': 'MLP',
            'survpath': 'SurvPath',
            'porpoise': 'PORPOISE',
            'mcat': 'MCAT',
            'motcat': 'MOTCAT',
            'pibd': 'PIBD',
            'mmp': 'MMP',
        }
        display = display_names.get(name, name)

        lines.append(
            f"  {display} & {mod_str} & {params_str} & {flops_str} & "
            f"{vram_str} & {train_str} & {infer_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    table_str = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(table_str)

    return table_str


# =============================================================================
# Main Profiling Logic
# =============================================================================

def profile_model(model_name, cfg, data_components, dataset, device, num_patients=30, num_warmup=5):
    """
    Profile a single model.

    Returns dict with all metrics.
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Profiling: {model_name}")
    logging.info(f"{'='*60}")

    # Override model name in config
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_copy.model.name = model_name

    reqs = get_model_requirements(model_name)
    modality = reqs.get('modality', 'unknown')

    # Configure branches for ProtoPathway
    if model_name == 'protopath':
        cfg_copy.model.branches.gene = True
        cfg_copy.model.branches.wsi = True

    # Build model kwargs
    model_kwargs = {
        'num_genes': data_components['graph_data']['num_genes'],
        'num_pathways': data_components['graph_data']['num_pathways']
    }

    # Compute centroids if needed
    if reqs.get('needs_centroids'):
        centroid_path = os.path.join(cfg.paths.processed_dir, 'centroids_fold_0.pt')
        if os.path.exists(centroid_path):
            model_kwargs['wsi_centroids'] = torch.load(centroid_path, weights_only=True)
            logging.info("  Loaded pre-computed centroids")
        else:
            logging.info("  Computing centroids...")
            patient_ids = list(data_components['wsi_features'].keys())[:100]
            wsi_subset = {pid: data_components['wsi_features'][pid] for pid in patient_ids}
            embeddings = sample_wsi_embeddings(wsi_subset, max_samples=50000)
            model_kwargs['wsi_centroids'] = compute_centroids(
                embeddings,
                n_clusters=cfg_copy.model.wsi_encoder.num_prototypes,
                seed=cfg.experiment.seed
            )

    # Build model
    try:
        model = build_model(cfg_copy, **model_kwargs)
    except Exception as e:
        logging.error(f"  Failed to build {model_name}: {e}")
        return None

    model = model.to(device)

    # --- 1. Parameter count ---
    trainable, total = count_parameters(model)
    logging.info(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # --- 2. Pre-load data ---
    batches, load_time = preload_batches(dataset, num_patients)

    if len(batches) == 0:
        logging.error(f"  No data available for {model_name}")
        return None

    # Get a sample batch for FLOPs/memory measurement
    sample_batch = batches[0].to(device)

    # --- 3. FLOPs ---
    builtin_flops = estimate_flops_builtin(model, sample_batch, cfg_copy, device)
    gnn_flops = estimate_gnn_flops(model, sample_batch, cfg_copy)
    total_flops = builtin_flops + gnn_flops
    logging.info(f"  FLOPs: {int(total_flops):,} (built-in: {int(builtin_flops):,}, GNN estimate: {int(gnn_flops):,})")

    # --- 4. Peak VRAM ---
    criterion = NLLSurvLoss() if cfg.task == 'survival' else nn.CrossEntropyLoss()
    peak_vram_train = measure_peak_memory(model, sample_batch, criterion, cfg_copy, device, mode='train')
    peak_vram_infer = measure_peak_memory(model, sample_batch, criterion, cfg_copy, device, mode='inference')
    logging.info(f"  Peak VRAM: {peak_vram_train:.0f} MB (train), {peak_vram_infer:.0f} MB (inference)")

    # --- 5. Training time ---
    # Rebuild model fresh (optimizer state from memory measurement is stale)
    model = build_model(cfg_copy, **model_kwargs).to(device)
    train_mean, train_std = time_training(model, batches, criterion, cfg_copy, device, num_warmup)
    logging.info(f"  Train time: {train_mean:.1f} ± {train_std:.1f} ms/patient")

    # --- 6. Inference time ---
    model = build_model(cfg_copy, **model_kwargs).to(device)
    infer_mean, infer_std = time_inference(model, batches, cfg_copy, device, num_warmup)
    logging.info(f"  Infer time: {infer_mean:.1f} ± {infer_std:.1f} ms/patient")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        'model': model_name,
        'modality': modality,
        'trainable_params': trainable,
        'total_params': total,
        'builtin_flops': builtin_flops,
        'gnn_flops': gnn_flops,
        'total_flops': total_flops,
        'peak_vram_train_mb': peak_vram_train,
        'peak_vram_infer_mb': peak_vram_infer,
        'train_ms_mean': train_mean,
        'train_ms_std': train_std,
        'infer_ms_mean': infer_mean,
        'infer_ms_std': infer_std,
        'load_time_s': load_time,
        'load_time_ms_per_patient': load_time / num_patients * 1000,
        'num_patients_profiled': len(batches),
        'num_warmup': num_warmup,
    }


def run(cfg, models_to_profile=None, num_patients=30, num_warmup=5):
    """
    Main profiling entry point.

    Args:
        cfg: OmegaConf config
        models_to_profile: List of model names to profile (None = all registered)
        num_patients: Number of patients to profile over
        num_warmup: Number of warmup iterations before timing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Profiling device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    logging.info("\nLoading dataset components...")
    data_components = load_dataset_components(cfg)

    # Create dataset from first fold
    splits = data_components['splits']
    fold_splits = splits['CV']['Fold 0']

    # Use validation set for profiling (smaller, avoids train-specific processing)
    from utils.dataset import MultimodalDataset
    dataset = MultimodalDataset(
        patient_ids=fold_splits['Val'],
        gene_expression_df=data_components['gene_expression_df'],
        graph_data=data_components['graph_data'],
        wsi_features=data_components['wsi_features'],
        labels_df=data_components['labels_df'],
        task=cfg.task,
        patient_id_col=cfg.patient_id_col
    )

    logging.info(f"Dataset: {len(dataset)} patients in fold 0 validation set")

    # Determine models to profile
    if models_to_profile is None:
        models_to_profile = get_available_models()
    logging.info(f"Models to profile: {models_to_profile}")

    # Profile each model
    all_results = []
    for model_name in models_to_profile:
        try:
            result = profile_model(
                model_name, cfg, data_components, dataset, device,
                num_patients=num_patients, num_warmup=num_warmup
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            logging.error(f"Failed to profile {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Output results ---
    output_dir = os.path.join(cfg.output.experiments_dir, 'profiling')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON output
    json_path = os.path.join(output_dir, f'profile_{cfg.dataset}_{timestamp}.json')
    meta = {
        'dataset': cfg.dataset,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'pytorch_version': torch.__version__,
        'num_patients_profiled': num_patients,
        'num_warmup': num_warmup,
        'timestamp': timestamp,
    }

    def sanitise(obj):
        if isinstance(obj, dict):
            return {k: sanitise(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitise(v) for v in obj]
        elif isinstance(obj, (torch.Tensor, np.integer, np.floating)):
            return obj.item() if hasattr(obj, 'item') else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = sanitise({'metadata': meta, 'results': all_results})
    with open(json_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    logging.info(f"\nJSON results saved to: {json_path}")

    # LaTeX output
    latex_path = os.path.join(output_dir, f'efficiency_table_{cfg.dataset}_{timestamp}.tex')
    table_str = generate_latex_table(all_results, latex_path)
    logging.info(f"LaTeX table saved to: {latex_path}")
    logging.info(f"\n{table_str}")

    # Summary table to console
    logging.info(f"\n{'='*90}")
    logging.info(f"{'Model':<12} {'Modality':<10} {'Params':<12} {'FLOPs':<12} "
                 f"{'VRAM(MB)':<10} {'Train(ms)':<12} {'Infer(ms)':<12}")
    logging.info(f"{'-'*90}")
    for r in all_results:
        logging.info(
            f"{r['model']:<12} {r['modality']:<10} {r['trainable_params']:<12,} "
            f"{int(r['total_flops']):<12,.0f} {r['peak_vram_train_mb']:<10.0f} "
            f"{r['train_ms_mean']:<12.1f} {r['infer_ms_mean']:<12.1f}"
        )
    logging.info(f"{'='*90}")

    return all_results