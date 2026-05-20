"""
Training Script for ProtoPathway.

Supports:
- Cross-validation and single train/test split
- Survival (NLLSurvLoss) and classification (CrossEntropy) tasks
- Multiple model types (ProtoPathway and baselines)
- Early stopping and learning rate scheduling
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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader as PyGDataLoader
from omegaconf import OmegaConf

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, accuracy_score

from utils.io import (
    get_model_schema,
    format_model_schema,
    setup_logging_to_file,
    build_experiment_name,
    save_source_snapshot
)

from utils.dataset import (
    load_dataset_components,
    create_fold_datasets,
    sample_wsi_embeddings,
    compute_centroids
)

from utils.survival import calculate_risk, debug_inversion
from utils.losses import NLLSurvLoss

# Use factory instead of direct import
from models.factory import build_model, get_model_requirements


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, cfg, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0

    # For survival metrics
    all_risks = []
    all_times = []
    all_events = []

    # For classification metrics
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if cfg.model.name == 'pibd':
            logits, aux_losses = model(batch)
        else:
            logits = model(batch, return_attention=False)

        if cfg.task == 'survival':

            target = batch.y['bin']
            time = batch.y['time']
            event = batch.y['event']

            if cfg.model.name == 'pibd':
                surv_loss = criterion(logits, target, time, event)
                loss = (surv_loss
                        + aux_losses['IB_loss']
                        + cfg.model.pibd.gamma * aux_losses['proxy_loss']
                        + cfg.model.pibd.sigma * (aux_losses['mimin'] + aux_losses['mimin_loss']))
            else:
                loss = criterion(logits, target, time, event)

            with torch.no_grad():
                risk = calculate_risk(logits)
                all_risks.append(risk.cpu())
                all_times.append(time.cpu())
                all_events.append(event.cpu())
        else:
            target = batch.y
            loss = criterion(logits, target)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        # L1 regularization
        if cfg.training.l1_lambda > 0:
            l1_loss = sum(p.abs().sum() for p in model.parameters())
            loss = loss + cfg.training.l1_lambda * l1_loss

        loss.backward()

        # Gradient clipping
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if cfg.task == 'survival':
        risks = torch.cat(all_risks).numpy()
        times = torch.cat(all_times).numpy()
        events = torch.cat(all_events).numpy()

        event_indicator = events.astype(bool) # event=1 → True (death)
        try:
            c_index = concordance_index_censored(event_indicator, times, risks)[0] # Expects True=uncensored=1
        except Exception as e:
            logging.warning(f"C-index failed: {e}")
            c_index = 0.5

        return {'loss': avg_loss, 'c_index': c_index}
    else:
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        acc = accuracy_score(targets, preds) * 100

        return {'loss': avg_loss, 'accuracy': acc}


@torch.no_grad()
def validate(model, loader, criterion, cfg, device):
    """Run validation."""
    model.eval()
    total_loss = 0.0

    all_risks = []
    all_times = []
    all_events = []
    all_probs = []
    all_targets = []
    all_patient_ids = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch, return_attention=False)

        if cfg.task == 'survival':
            target = batch.y['bin']
            time = batch.y['time']
            event = batch.y['event']

            loss = criterion(logits, target, time, event)

            risk = calculate_risk(logits)
            all_risks.append(risk.cpu())
            all_times.append(time.cpu())
            all_events.append(event.cpu())
        else:
            target = batch.y
            loss = criterion(logits, target)

            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())

        total_loss += loss.item()

        if hasattr(batch, 'patient_id'):
            all_patient_ids.append(batch.patient_id)

    avg_loss = total_loss / len(loader)


    if cfg.task == 'survival':
        risks = torch.cat(all_risks).numpy()
        times = torch.cat(all_times).numpy()
        events = torch.cat(all_events).numpy()

        if cfg.training.get("debug_survival", False):

            debug_inversion(risks, times, events)

            risk_std = np.std(risks)
            event_rate = events.mean()
            corr = np.corrcoef(risks, times)[0, 1]

            spread = np.percentile(risks, [5, 25, 50, 75, 95])

            logging.info(
                f"[Survival Debug] "
                f"risk_std={risk_std:.4f} | "
                f"event_rate={event_rate:.3f} | "
                f"risk_time_corr={corr:.3f} | "
                f"spread(p5-p95)={spread}"
            )

        event_indicator = events.astype(bool)
        try:
            c_index = concordance_index_censored(event_indicator, times, risks)[0]
        except:
            c_index = 0.5

        return {
            'loss': avg_loss,
            'c_index': c_index,
            'risks': risks,
            'times': times,
            'events': events,
            'patient_ids': all_patient_ids
        }
    else:
        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()
        preds = probs.argmax(axis=1)

        acc = accuracy_score(targets, preds) * 100

        # AUC
        if probs.shape[1] == 2:
            auc = roc_auc_score(targets, probs[:, 1])
        else:
            try:
                auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            except:
                auc = 0.5

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'auc': auc,
            'probs': probs,
            'targets': targets,
            'patient_ids': all_patient_ids
        }


def train_fold(
    fold_idx,
    train_dataset,
    val_dataset,
    data_components,
    cfg,
    device,
    output_dir
):
    """Train a single fold."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training Fold {fold_idx}")
    logging.info(f"{'='*60}")
    logging.info(f"  Train: {len(train_dataset)} patients")
    logging.info(f"  Val: {len(val_dataset)} patients")

    # Data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )

    # Get model requirements
    model_reqs = get_model_requirements(cfg.model.name)

    # Prepare kwargs for model building
    model_kwargs = {
        'num_genes': data_components['graph_data']['num_genes'],
        'num_pathways': data_components['graph_data']['num_pathways']
    }

    # Compute or load centroids for models that need them
    centroids = None
    if model_reqs.get('needs_centroids') and cfg.model.get('branches', {}).get('wsi', False):
        if cfg.model.wsi_encoder.use_precomputed_centroids:
            K = cfg.model.wsi_encoder.num_prototypes
            centroid_dir = os.path.join(cfg.paths.processed_dir, 'centroids', f'K{K}')
            os.makedirs(centroid_dir, exist_ok=True)
            centroid_path = os.path.join(centroid_dir, f'centroids_fold_{fold_idx}.pt')

            if os.path.exists(centroid_path):
                logging.info(f"Loading pre-computed centroids from {centroid_path}")
                centroids = torch.load(centroid_path, weights_only=True)
            else:
                logging.info("Computing centroids from training data...")
                # Sample from training patients only
                train_wsi = {pid: data_components['wsi_features'][pid]
                            for pid in train_dataset.patient_ids
                            if pid in data_components['wsi_features']}
                embeddings = sample_wsi_embeddings(train_wsi)
                centroids = compute_centroids(
                    embeddings,
                    n_clusters=cfg.model.wsi_encoder.num_prototypes,
                    seed=cfg.experiment.seed
                )
                torch.save(centroids, centroid_path)
                logging.info(f"Saved centroids to {centroid_path}")

    model_kwargs['wsi_centroids'] = centroids

    # Build model using factory
    model = build_model(cfg, **model_kwargs)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Model: {cfg.model.name}")
    logging.info(f"  Parameters: {n_params:,}")

    # Generate and save model schema (only for first fold to avoid duplicates)
    if fold_idx == 0 or not os.path.exists(os.path.join(output_dir, 'model_schema.txt')):
        schema = get_model_schema(model, cfg)

        # Save formatted text version
        schema_text = format_model_schema(schema)
        schema_txt_path = os.path.join(output_dir, 'model_schema.txt')
        with open(schema_txt_path, 'w') as f:
            f.write(schema_text)

        logging.info(f"  Model schema saved to {schema_txt_path}")
        logging.info(schema_text)

    # Loss function
    if cfg.task == 'survival':
        criterion = NLLSurvLoss(alpha=cfg.training.nll_alpha)
    else:
        criterion = nn.CrossEntropyLoss()

    if cfg.model.name == 'protopath':
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay)
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )


    # Scheduler
    scheduler = None
    if cfg.training.scheduler.type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs)
    elif cfg.training.scheduler.type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=cfg.training.scheduler.step_size,
            gamma=cfg.training.scheduler.gamma
        )
    elif cfg.training.scheduler.type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.monitor_mode,
            factor=cfg.training.scheduler.factor,
            patience=cfg.training.scheduler.plateau_patience
        )

    # Training loop
    best_metric = -float('inf') if cfg.training.monitor_mode == 'max' else float('inf')
    best_epoch = 0
    patience_counter = 0

    history = {'train': [], 'val': []}

    for epoch in range(cfg.training.max_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, cfg, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, cfg, device)

        # Get current metric
        current_metric = val_metrics[cfg.training.monitor_metric]

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric)
            else:
                scheduler.step()

        # Check for improvement
        improved = False
        if cfg.training.monitor_mode == 'max':
            improved = current_metric > best_metric
        else:
            improved = current_metric < best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint_path = os.path.join(output_dir, f'best_fold_{fold_idx}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'cfg': OmegaConf.to_container(cfg)
            }, checkpoint_path)
        else:
            patience_counter += 1

        # Store history (without large arrays)
        history['train'].append({k: v for k, v in train_metrics.items()})
        history['val'].append({
            k: v for k, v in val_metrics.items()
            if k not in ['risks', 'times', 'events', 'probs', 'targets', 'patient_ids']
        })

        epoch_time = time.time() - epoch_start

        # Logging
        if cfg.task == 'survival':
            logging.info(
                f"Epoch {epoch+1:3d}/{cfg.training.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, C-idx: {train_metrics['c_index']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, C-idx: {val_metrics['c_index']:.4f} | "
                f"Best: {best_metric:.4f} (ep {best_epoch+1}) | "
                f"Time: {epoch_time:.1f}s"
            )
        else:
            logging.info(
                f"Epoch {epoch+1:3d}/{cfg.training.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, AUC: {val_metrics['auc']:.4f} | "
                f"Best: {best_metric:.4f} (ep {best_epoch+1}) | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping
        if patience_counter >= cfg.training.patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for final validation
    checkpoint = torch.load(os.path.join(output_dir, f'best_fold_{fold_idx}.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_val_metrics = validate(model, val_loader, criterion, cfg, device)

    del model, optimizer, criterion
    if scheduler is not None:
        del scheduler
    del train_loader, val_loader
    torch.cuda.empty_cache()

    return {
        'fold_idx': fold_idx,
        'best_epoch': best_epoch,
        'best_metric': best_metric,
        'history': history,
        'final_val_metrics': final_val_metrics
    }


def run(cfg):
    """Main training entry point."""
    # Setup
    set_seed(cfg.experiment.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = build_experiment_name(cfg, timestamp)
    output_dir = os.path.join(cfg.output.experiments_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # console log
    log_path, file_handler = setup_logging_to_file(output_dir)

    save_source_snapshot(output_dir, cfg.paths.code_dir)

    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.info(f"Experiment directory: {output_dir}")

    # Load data
    data_components = load_dataset_components(cfg)
    splits = data_components['splits']

    # Run training
    fold_results = []

    if cfg.cv.enabled:
        # Cross-validation
        for fold_name, fold_splits in splits['CV'].items():
            fold_idx = int(fold_name.split()[-1])

            train_dataset, val_dataset = create_fold_datasets(
                data_components, fold_splits, cfg
            )

            result = train_fold(
                fold_idx=fold_idx,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_components=data_components,
                cfg=cfg,
                device=device,
                output_dir=output_dir
            )

            fold_results.append(result)
            del train_dataset, val_dataset
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    else:
        # Single train/test split
        if len(splits['Test']) > 0:
            fold_splits = {
                'Train': splits['Train'],
                'Val': splits['Test']
            }
        else:
            # Use first CV fold
            first_fold = list(splits['CV'].values())[0]
            fold_splits = first_fold

        train_dataset, val_dataset = create_fold_datasets(
            data_components, fold_splits, cfg
        )

        result = train_fold(
            fold_idx=0,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_components=data_components,
            cfg=cfg,
            device=device,
            output_dir=output_dir
        )

        fold_results.append(result)

    # Aggregate results
    logging.info("\n" + "="*60)
    logging.info("Training Complete - Summary")
    logging.info("="*60)

    metrics = [r['best_metric'] for r in fold_results]
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)

    metric_name = cfg.training.monitor_metric
    logging.info(f"  {metric_name}: {mean_metric:.4f} ± {std_metric:.4f}")

    for r in fold_results:
        logging.info(f"    Fold {r['fold_idx']}: {r['best_metric']:.4f} (epoch {r['best_epoch']+1})")

    # Save summary
    summary = {
        'experiment_name': exp_name,
        'dataset': cfg.dataset,
        'model': cfg.model.name,
        'task': cfg.task,
        f'mean_{metric_name}': float(mean_metric),
        f'std_{metric_name}': float(std_metric),
        'fold_results': [
            {
                'fold_idx': r['fold_idx'],
                'best_epoch': r['best_epoch'],
                'best_metric': float(r['best_metric'])
            }
            for r in fold_results
        ]
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\nResults saved to {output_dir}")
    logging.info(f"Training log saved to {log_path}")

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return fold_results