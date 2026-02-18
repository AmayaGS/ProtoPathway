"""
Prototype importance shift visualization.

Compares raw WSI gate importance (Signal E) with pathway-attended
importance (Signal H), revealing how biological context reshapes
the model's morphological priorities.

Visualizes the PrototypeShiftAnalyzer results.
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

COLOR_GAINED = '#E53935'   # Red: gained importance after pathway attention
COLOR_LOST = '#2196F3'     # Blue: lost importance
COLOR_LOW = '#64B5F6'      # Light blue for low risk
COLOR_HIGH = '#EF9A9A'     # Light red for high risk


def plot_prototype_shift(
    shift_analyzer,
    output_path: str,
    title: str = 'Prototype Importance Shift After Pathway Attention',
    show_by_risk: bool = True,
    figsize: tuple = (12, 6),
):
    """
    Visualize prototype importance shift (Signal E → Signal H).

    Creates a grouped bar chart showing before/after importance for
    each prototype, with shift direction indicated by color.

    Args:
        shift_analyzer: PrototypeShiftAnalyzer instance.
        output_path: Path to save figure.
        title: Plot title.
        show_by_risk: If True, create additional per-risk-group panel.
        figsize: Figure size.
    """
    # Overall shift
    shift_df = shift_analyzer.analyze_shift()
    _plot_shift_bars(shift_df, title, output_path, figsize)

    # Per-risk-group comparison
    if show_by_risk:
        by_risk = shift_analyzer.analyze_shift_by_risk()
        if 'low' in by_risk and 'high' in by_risk:
            risk_path = output_path.replace('.pdf', '_by_risk.pdf')
            risk_path = _ensure_pdf(risk_path)
            _plot_shift_comparison(
                by_risk['low'], by_risk['high'],
                f'{title}\n(By Risk Group)',
                risk_path, figsize
            )


def _plot_shift_bars(
    shift_df: pd.DataFrame,
    title: str,
    output_path: str,
    figsize: tuple,
):
    """Paired bar chart: before (E) and after (H) for each prototype."""
    n = len(shift_df)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n)
    width = 0.35

    # Sort by shift magnitude for visual clarity
    shift_df = shift_df.sort_values('mean_shift', ascending=True).reset_index(drop=True)

    bars_e = ax.bar(
        x - width / 2, shift_df['mean_wsi_gate'],
        width, label='Before (WSI Gate)',
        color='#90CAF9', edgecolor='white', linewidth=0.5
    )
    bars_h = ax.bar(
        x + width / 2, shift_df['mean_fusion_gate'],
        width, label='After (Pathway-Attended)',
        color='#EF9A9A', edgecolor='white', linewidth=0.5
    )

    # Significance markers
    for i, row in shift_df.iterrows():
        if row.get('significant', False):
            y_max = max(row['mean_wsi_gate'], row['mean_fusion_gate'])
            ax.text(
                i, y_max + 0.005, '★',
                ha='center', va='bottom', fontsize=12, color='#FF6F00'
            )

    # Shift arrows
    for i, row in shift_df.iterrows():
        e_val = row['mean_wsi_gate']
        h_val = row['mean_fusion_gate']
        mid_y = (e_val + h_val) / 2
        color = COLOR_GAINED if h_val > e_val else COLOR_LOST
        ax.annotate(
            '', xy=(i + width / 2, h_val),
            xytext=(i - width / 2, e_val),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=1.5, alpha=0.7
            )
        )

    ax.set_xticks(x)
    ax.set_xticklabels(shift_df['prototype'], fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('Mean Gate Weight', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved shift plot to {output_path}")


def _plot_shift_comparison(
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    title: str,
    output_path: str,
    figsize: tuple,
):
    """
    Side-by-side shift comparison for low vs high risk groups.

    Shows whether pathway context changes prototype priorities
    differently depending on predicted risk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.2, figsize[1]),
                             sharey=True)

    for ax, df, group_title, color_before, color_after in [
        (axes[0], low_df, 'Low Risk', '#BBDEFB', '#64B5F6'),
        (axes[1], high_df, 'High Risk', '#FFCDD2', '#EF5350'),
    ]:
        df = df.sort_values('mean_shift', ascending=True).reset_index(drop=True)
        n = len(df)
        x = np.arange(n)
        width = 0.35

        ax.bar(
            x - width / 2, df['mean_wsi_gate'], width,
            label='Before (WSI Gate)', color=color_before,
            edgecolor='white', linewidth=0.5
        )
        ax.bar(
            x + width / 2, df['mean_fusion_gate'], width,
            label='After (Pathway-Attended)', color=color_after,
            edgecolor='white', linewidth=0.5
        )

        for i, row in df.iterrows():
            if row.get('significant', False):
                y_max = max(row['mean_wsi_gate'], row['mean_fusion_gate'])
                ax.text(
                    i, y_max + 0.005, '★',
                    ha='center', va='bottom', fontsize=11, color='#FF6F00'
                )

        ax.set_xticks(x)
        ax.set_xticklabels(df['prototype'], fontsize=8, rotation=45, ha='right')
        ax.set_title(group_title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Mean Gate Weight', fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved shift comparison to {output_path}")


def plot_shift_slope(
    shift_analyzer,
    output_path: str,
    title: str = 'Prototype Importance: Before vs After Pathway Attention',
    figsize: tuple = (8, 6),
):
    """
    Slope chart (slopegraph) showing E→H shift per prototype.

    Each line connects a prototype's before (E) and after (H) importance,
    colored by direction. Gives an immediate visual of which prototypes
    gain/lose importance from pathway context.
    """
    shift_df = shift_analyzer.analyze_shift()
    n = len(shift_df)

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in shift_df.iterrows():
        color = COLOR_GAINED if row['mean_shift'] > 0 else COLOR_LOST
        lw = 2.5 if row.get('significant', False) else 1.2
        alpha = 0.9 if row.get('significant', False) else 0.5

        ax.plot(
            [0, 1],
            [row['mean_wsi_gate'], row['mean_fusion_gate']],
            color=color, linewidth=lw, alpha=alpha,
            marker='o', markersize=6
        )

        # Label on the right side
        ax.text(
            1.05, row['mean_fusion_gate'],
            row['prototype'],
            va='center', fontsize=8, color=color,
            fontweight='bold' if row.get('significant', False) else 'normal'
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ['Before\n(WSI Gate)', 'After\n(Pathway-Attended)'],
        fontsize=11
    )
    ax.set_ylabel('Mean Gate Weight', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(-0.2, 1.5)

    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_GAINED, lw=2,
                    label='Gained importance'),
        plt.Line2D([0], [0], color=COLOR_LOST, lw=2,
                    label='Lost importance'),
        plt.Line2D([0], [0], color='grey', lw=2.5,
                    label='Significant (FDR)'),
        plt.Line2D([0], [0], color='grey', lw=1.2, alpha=0.5,
                    label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved slope plot to {output_path}")


def _ensure_pdf(path):
    base, ext = os.path.splitext(path)
    if ext.lower() != '.pdf':
        return base + '.pdf'
    return path