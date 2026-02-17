"""
Visualization Script for ProtoPathway.

Generates visualizations from evaluation results:
- Kaplan-Meier survival curves
- Risk stratification plots
- Attention heatmaps
- Prototype-pathway interaction maps
- Gene importance rankings
"""

import os
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


def plot_kaplan_meier(times, events, risk_scores, output_path, n_groups=2):
    """
    Plot Kaplan-Meier survival curves stratified by risk.

    Args:
        times: Survival times
        events: Event indicators
        risk_scores: Predicted risk scores
        output_path: Path to save figure
        n_groups: Number of risk groups (2 or 4)
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logging.warning("lifelines not installed, skipping KM plot")
        return

    # Stratify into risk groups
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

    # Log-rank test for 2 groups
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


def plot_attention_heatmap(attention_matrix, row_labels, col_labels, output_path, title='Attention Weights'):
    """
    Plot attention heatmap.

    Args:
        attention_matrix: 2D numpy array
        row_labels: Labels for rows
        col_labels: Labels for columns
        output_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')

    # Limit labels for readability
    max_labels = 30
    if len(row_labels) > max_labels:
        step = len(row_labels) // max_labels
        ax.set_yticks(np.arange(0, len(row_labels), step))
        ax.set_yticklabels([row_labels[i] for i in range(0, len(row_labels), step)], fontsize=8)
    else:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

    if len(col_labels) > max_labels:
        step = len(col_labels) // max_labels
        ax.set_xticks(np.arange(0, len(col_labels), step))
        ax.set_xticklabels([col_labels[i] for i in range(0, len(col_labels), step)],
                           fontsize=8, rotation=45, ha='right')
    else:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)

    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved attention heatmap to {output_path}")


def run(results_dir, output_dir=None):
    """
    Generate visualizations from evaluation results.

    Args:
        results_dir: Path to experiment results directory
        output_dir: Output directory for figures (defaults to results_dir/figures)
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / 'figures'
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Generating visualizations from {results_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Load evaluation results
    eval_dir = results_dir / 'evaluation'

    # Check for predictions file
    pred_files = list(eval_dir.glob('predictions*.csv'))
    if not pred_files:
        logging.warning("No predictions file found")
        return

    predictions = pd.read_csv(pred_files[0])

    # Check task type
    if 'risk_score' in predictions.columns:
        # Survival task
        logging.info("Generating survival visualizations...")

        plot_kaplan_meier(
            predictions['survival_time'].values,
            predictions['event'].values,
            predictions['risk_score'].values,
            output_dir / 'kaplan_meier.pdf'
        )

        plot_risk_distribution(
            predictions['risk_score'].values,
            predictions['event'].values,
            output_dir / 'risk_distribution.pdf'
        )
    else:
        # Classification task
        logging.info("Generating classification visualizations...")
        # Add classification plots here

    # Load attention weights if available
    attention_files = list(eval_dir.glob('attention*/attention_weights.pkl'))
    if attention_files:
        logging.info("Processing attention weights...")

        with open(attention_files[0], 'rb') as f:
            attention_data = pickle.load(f)

        # Generate attention visualizations
        # This depends on what attention outputs are available

        attention_outputs = attention_data.get('attention_outputs', {})

        if 'cross_modal_attention' in attention_outputs:
            logging.info("  Cross-modal attention found")
            # Average across patients
            attn_list = attention_outputs['cross_modal_attention']
            if len(attn_list) > 0 and isinstance(attn_list[0], torch.Tensor):
                import torch
                avg_attn = torch.stack(attn_list).mean(dim=0).numpy()

                plot_attention_heatmap(
                    avg_attn,
                    row_labels=[f'Proto {i}' for i in range(avg_attn.shape[0])],
                    col_labels=[f'Pathway {i}' for i in range(avg_attn.shape[1])],
                    output_path=output_dir / 'cross_modal_attention.pdf',
                    title='Prototype-Pathway Cross Attention (Averaged)'
                )

    logging.info(f"Visualizations saved to {output_dir}")