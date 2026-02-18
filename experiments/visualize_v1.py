"""
Visualization Script for ProtoPathway.

Generates visualizations from evaluation results:
- Kaplan-Meier survival curves
- Risk stratification plots
- Gene-pathway attention heatmaps
- Pathway importance rankings
- Prototype assignment distributions
- WSI vs fusion gate comparison
- Cross-modal prototype-pathway attention
- Per-patient gene importance
"""

import os
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


# ============================================================
# Survival Plots
# ============================================================

def plot_kaplan_meier(times, events, risk_scores, output_path, n_groups=2):
    """Plot Kaplan-Meier survival curves stratified by risk."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logging.warning("lifelines not installed, skipping KM plot")
        return

    if n_groups == 2:
        threshold = np.median(risk_scores)
        groups = (risk_scores > threshold).astype(int)
        group_names = ['Low Risk', 'High Risk']
        colors = ['green', 'red']
    else:
        percentiles = np.percentile(risk_scores, [25, 50, 75])
        groups = np.digitize(risk_scores, percentiles)
        group_names = ['Very Low', 'Low', 'High', 'Very High']
        colors = ['darkgreen', 'lightgreen', 'orange', 'red']

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for i, (name, color) in enumerate(zip(group_names, colors)):
        mask = groups == i
        if mask.sum() > 0:
            kmf.fit(times[mask], events[mask], label=f"{name} (n={mask.sum()})")
            kmf.plot(ax=ax, ci_show=True, color=color)

    if n_groups == 2:
        result = logrank_test(
            times[groups == 0], times[groups == 1],
            events[groups == 0], events[groups == 1]
        )
        ax.text(0.02, 0.02, f'Log-rank p={result.p_value:.4f}',
                transform=ax.transAxes, fontsize=10)

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved KM plot to {output_path}")


def plot_risk_distribution(risk_scores, events, output_path):
    """Plot distribution of risk scores by event status."""
    fig, ax = plt.subplots(figsize=(8, 5))
    event_mask = events.astype(bool)

    ax.hist(risk_scores[event_mask], bins=30, alpha=0.6,
            label=f'Event (n={event_mask.sum()})', color='red')
    ax.hist(risk_scores[~event_mask], bins=30, alpha=0.6,
            label=f'Censored (n={(~event_mask).sum()})', color='blue')

    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Risk Score Distribution by Event Status', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved risk distribution to {output_path}")


# ============================================================
# Attention Heatmaps
# ============================================================

def plot_attention_heatmap(attention_matrix, row_labels, col_labels, output_path,
                           title='Attention Weights', cmap='viridis'):
    """Plot attention heatmap with automatic label thinning."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto')

    max_labels = 30
    for axis, labels, setter, rotation in [
        ('y', row_labels, ax.set_yticklabels, 0),
        ('x', col_labels, ax.set_xticklabels, 45)
    ]:
        if len(labels) > max_labels:
            step = len(labels) // max_labels
            ticks = np.arange(0, len(labels), step)
            getattr(ax, f'set_{axis}ticks')(ticks)
            setter([labels[i] for i in ticks], fontsize=8,
                   rotation=rotation, ha='right' if rotation else 'center')
        else:
            getattr(ax, f'set_{axis}ticks')(np.arange(len(labels)))
            setter(labels, fontsize=8,
                   rotation=rotation, ha='right' if rotation else 'center')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved attention heatmap to {output_path}")


# ============================================================
# Gene & Pathway Visualizations
# ============================================================

def plot_pathway_importance(pathway_importance, output_path, pathway_names=None, top_k=20):
    """
    Bar chart of pathway importance from gating mechanism.

    Args:
        pathway_importance: [num_pathways] averaged across patients
        pathway_names: Optional list of pathway names
        top_k: Number of top pathways to show
    """
    if pathway_names is None:
        pathway_names = [f'Pathway {i}' for i in range(len(pathway_importance))]

    # Sort by importance
    sorted_idx = np.argsort(pathway_importance)[::-1][:top_k]
    sorted_names = [pathway_names[i] for i in sorted_idx]
    sorted_values = pathway_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    bars = ax.barh(range(top_k), sorted_values[::-1], color='steelblue')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Gate Weight (softmax)', fontsize=12)
    ax.set_title(f'Top {top_k} Pathways by Importance', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved pathway importance to {output_path}")


def plot_gene_importance(gene_pathway_attention, pathway_importance, output_path,
                         gene_names=None, top_k=30):
    """
    Bar chart of overall gene importance (attention × pathway gate).

    Args:
        gene_pathway_attention: [num_genes, num_pathways] averaged across patients
        pathway_importance: [num_pathways] averaged across patients
        gene_names: Optional list of gene names
        top_k: Number of top genes to show
    """
    # Weighted sum: gene importance = sum over pathways of (attn * gate_weight)
    gene_importance = (gene_pathway_attention * pathway_importance[np.newaxis, :]).sum(axis=1)

    if gene_names is None:
        gene_names = [f'Gene {i}' for i in range(len(gene_importance))]

    sorted_idx = np.argsort(gene_importance)[::-1][:top_k]
    sorted_names = [gene_names[i] for i in sorted_idx]
    sorted_values = gene_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    ax.barh(range(top_k), sorted_values[::-1], color='coral')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel('Importance (attention × pathway gate)', fontsize=12)
    ax.set_title(f'Top {top_k} Genes by Importance', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved gene importance to {output_path}")


# ============================================================
# Prototype Visualizations
# ============================================================

def plot_prototype_assignment_distribution(patch_assignments_list, output_path, num_prototypes=None):
    """
    Histogram of how patches distribute across prototypes (averaged across patients).

    Args:
        patch_assignments_list: List of per-patient assignment dicts
        num_prototypes: Number of prototypes (inferred if not given)
    """
    all_hard = []
    for pa in patch_assignments_list:
        if isinstance(pa, dict) and 'hard_assignments' in pa:
            hard = pa['hard_assignments']
            if isinstance(hard, torch.Tensor):
                hard = hard.numpy()
            all_hard.append(hard)

    if not all_hard:
        logging.warning("No hard assignments found, skipping prototype distribution")
        return

    if num_prototypes is None:
        num_prototypes = max(h.max() for h in all_hard) + 1

    # Count per prototype across all patients
    counts = np.zeros(num_prototypes)
    for hard in all_hard:
        for i in range(num_prototypes):
            counts[i] += (hard == i).sum()

    # Normalize to average per patient
    counts /= len(all_hard)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(num_prototypes), counts, color='mediumpurple')
    ax.set_xlabel('Prototype Index', fontsize=12)
    ax.set_ylabel('Avg Patches Assigned', fontsize=12)
    ax.set_title('Prototype Utilization (Hard Assignments)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved prototype distribution to {output_path}")


def plot_gate_comparison(wsi_gates_list, fusion_gates_list, output_path, num_prototypes=None):
    """
    Side-by-side comparison of WSI encoder gates vs fusion gates per prototype.

    Shows how pathway context changes prototype importance.

    Args:
        wsi_gates_list: List of per-patient WSI gate weight tensors
        fusion_gates_list: List of per-patient fusion gate weight tensors
    """
    def _average_gates(gates_list):
        tensors = []
        for g in gates_list:
            if isinstance(g, torch.Tensor):
                tensors.append(g.squeeze().numpy())
            elif isinstance(g, dict) and 'gate_weights' in g:
                t = g['gate_weights']
                if isinstance(t, torch.Tensor):
                    tensors.append(t.squeeze().numpy())
        if not tensors:
            return None
        return np.stack(tensors).mean(axis=0)

    wsi_avg = _average_gates(wsi_gates_list)
    fusion_avg = _average_gates(fusion_gates_list)

    if wsi_avg is None or fusion_avg is None:
        logging.warning("Missing gate data for comparison, skipping")
        return

    num_prototypes = len(wsi_avg)
    x = np.arange(num_prototypes)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Side by side bars
    axes[0].bar(x - width/2, wsi_avg, width, label='WSI Gates', color='steelblue')
    axes[0].bar(x + width/2, fusion_avg, width, label='Fusion Gates', color='coral')
    axes[0].set_xlabel('Prototype Index', fontsize=12)
    axes[0].set_ylabel('Gate Weight', fontsize=12)
    axes[0].set_title('Prototype Gating: WSI vs Fusion', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    # Difference plot
    diff = fusion_avg - wsi_avg
    colors = ['coral' if d > 0 else 'steelblue' for d in diff]
    axes[1].bar(x, diff, color=colors)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Prototype Index', fontsize=12)
    axes[1].set_ylabel('Fusion - WSI Gate Δ', fontsize=12)
    axes[1].set_title('Pathway Context Effect on Prototype Importance', fontsize=14)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved gate comparison to {output_path}")


# ============================================================
# Cross-Modal Attention
# ============================================================

def plot_cross_modal_attention(attn_list, output_path, pathway_names=None):
    """
    Averaged prototype-pathway cross attention heatmap.

    Args:
        attn_list: List of per-patient [num_prototypes, num_pathways] tensors
    """
    tensors = [a.numpy() if isinstance(a, torch.Tensor) else a for a in attn_list]
    avg_attn = np.stack(tensors).mean(axis=0)

    num_proto, num_path = avg_attn.shape
    row_labels = [f'Proto {i}' for i in range(num_proto)]
    col_labels = pathway_names if pathway_names else [f'Pathway {i}' for i in range(num_path)]

    plot_attention_heatmap(
        avg_attn, row_labels, col_labels, output_path,
        title='Prototype → Pathway Cross Attention (Averaged)',
        cmap='magma'
    )


# ============================================================
# Main Entry Point
# ============================================================

def run(results_dir, output_dir=None, pathway_names=None, gene_names=None):
    """
    Generate all visualizations from evaluation results.

    Args:
        results_dir: Path to experiment directory
        output_dir: Output directory for figures (defaults to results_dir/figures)
        pathway_names: Optional list of pathway names for labeling
        gene_names: Optional list of gene names for labeling
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / 'figures'
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Generating visualizations from {results_dir}")
    logging.info(f"Output directory: {output_dir}")

    eval_dir = results_dir / 'evaluation'

    # ----------------------------------------------------------
    # Survival / Classification Plots
    # ----------------------------------------------------------
    pred_files = list(eval_dir.glob('predictions*.csv'))
    if not pred_files:
        logging.warning("No predictions files found")
    else:
        # Use first fold or aggregate — iterate all for per-fold plots
        for pred_file in pred_files:
            predictions = pd.read_csv(pred_file)
            suffix = pred_file.stem.replace('predictions', '')  # e.g. '_fold_0'

            if 'risk_score' in predictions.columns:
                logging.info(f"Generating survival plots{suffix}...")

                plot_kaplan_meier(
                    predictions['survival_time'].values,
                    predictions['event'].values,
                    predictions['risk_score'].values,
                    output_dir / f'kaplan_meier{suffix}.pdf'
                )

                plot_risk_distribution(
                    predictions['risk_score'].values,
                    predictions['event'].values,
                    output_dir / f'risk_distribution{suffix}.pdf'
                )
            else:
                logging.info(f"Generating classification plots{suffix}...")
                # TODO: add confusion matrix plot, ROC curves

    # ----------------------------------------------------------
    # Attention Visualizations
    # ----------------------------------------------------------
    attention_files = list(eval_dir.glob('attention*/attention_weights.pkl'))

    for attn_file in attention_files:
        # Parse fold suffix from directory name
        fold_suffix = attn_file.parent.name.replace('attention', '')  # e.g. '_fold_0'
        logging.info(f"Processing attention weights{fold_suffix}...")

        with open(attn_file, 'rb') as f:
            attention_data = pickle.load(f)

        patient_ids = attention_data.get('patient_ids', [])
        outputs = attention_data.get('attention_outputs', {})

        # 1. Gene-pathway attention heatmap
        if 'gene_pathway_attention' in outputs:
            attn_list = outputs['gene_pathway_attention']
            tensors = [a.numpy() if isinstance(a, torch.Tensor) else a for a in attn_list]
            avg_gene_pathway = np.stack(tensors).mean(axis=0)

            row_labels = gene_names if gene_names else [f'Gene {i}' for i in range(avg_gene_pathway.shape[0])]
            col_labels = pathway_names if pathway_names else [f'Pathway {i}' for i in range(avg_gene_pathway.shape[1])]

            plot_attention_heatmap(
                avg_gene_pathway, row_labels, col_labels,
                output_dir / f'gene_pathway_attention{fold_suffix}.pdf',
                title='Gene-Pathway GATv2 Attention (Averaged)',
                cmap='viridis'
            )
            logging.info("  ✓ Gene-pathway attention heatmap")

        # 2. Pathway importance
        if 'pathway_importance' in outputs:
            pi_list = outputs['pathway_importance']
            tensors = [p.numpy() if isinstance(p, torch.Tensor) else p for p in pi_list]
            avg_pathway_imp = np.stack(tensors).mean(axis=0)

            plot_pathway_importance(
                avg_pathway_imp,
                output_dir / f'pathway_importance{fold_suffix}.pdf',
                pathway_names=pathway_names
            )
            logging.info("  ✓ Pathway importance ranking")

            # 3. Gene importance (requires both gene-pathway attention and pathway importance)
            if 'gene_pathway_attention' in outputs:
                plot_gene_importance(
                    avg_gene_pathway, avg_pathway_imp,
                    output_dir / f'gene_importance{fold_suffix}.pdf',
                    gene_names=gene_names
                )
                logging.info("  ✓ Gene importance ranking")

        # 4. Prototype assignment distribution
        if 'patch_assignments' in outputs:
            plot_prototype_assignment_distribution(
                outputs['patch_assignments'],
                output_dir / f'prototype_utilization{fold_suffix}.pdf'
            )
            logging.info("  ✓ Prototype assignment distribution")

        # 5. WSI vs Fusion gate comparison
        if 'patch_assignments' in outputs and 'fusion_gate_weights' in outputs:
            plot_gate_comparison(
                outputs['patch_assignments'],  # contains WSI gate_weights
                outputs['fusion_gate_weights'],
                output_dir / f'gate_comparison{fold_suffix}.pdf'
            )
            logging.info("  ✓ WSI vs Fusion gate comparison")

        # 6. Cross-modal attention
        if 'cross_modal_attention' in outputs:
            plot_cross_modal_attention(
                outputs['cross_modal_attention'],
                output_dir / f'cross_modal_attention{fold_suffix}.pdf',
                pathway_names=pathway_names
            )
            logging.info("  ✓ Cross-modal attention heatmap")

    logging.info(f"All visualizations saved to {output_dir}")