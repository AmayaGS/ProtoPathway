"""
Rank-based bar plots for differential importance analysis.

Produces horizontal bar charts showing:
    - Rank difference between risk groups (directional, colored by group)
    - Rank-biserial correlation (effect size)
    - FDR significance markers

Consumes CSVs from ImportanceAnalyzer.save_results().
"""

import os
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Consistent palette
COLOR_LOW = '#2196F3'   # Blue for Low Risk
COLOR_HIGH = '#E53935'  # Red for High Risk
COLOR_NS = '#BDBDBD'    # Grey for non-significant


def plot_rank_difference_bars(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    n: int = 30,
    entity_col: str = 'entity',
    metric_col: str = 'rank_difference',
    sig_col: str = 'significant',
    effect_col: str = 'rank_biserial_r',
    fdr_col: str = 'p_value_fdr',
    group_names: dict = None,
    figsize_per_row: float = 0.35,
    min_fig_height: float = 6,
    annotate_effect_size: bool = True,
    annotate_fdr: bool = True,
):
    """
    Horizontal bar chart of rank differences between risk groups.

    Bars are colored by direction: blue for Low Risk dominant,
    red for High Risk dominant. Grey if not significant.

    Args:
        df: DataFrame from ImportanceAnalyzer rank analysis.
        title: Plot title.
        output_path: Path to save (PDF).
        n: Maximum number of entities to show.
        entity_col: Column with entity names.
        metric_col: Column to plot on x-axis.
        sig_col: Column for significance flag.
        effect_col: Column for effect size annotation.
        fdr_col: Column for FDR p-value annotation.
        group_names: Dict with display names, e.g. {'low': 'Low Risk'}.
        figsize_per_row: Height per bar row.
        min_fig_height: Minimum figure height.
        annotate_effect_size: Show r_rb values on bars.
        annotate_fdr: Show FDR p-values on bars.
    """
    group_names = group_names or {'low': 'Low Risk', 'high': 'High Risk'}

    # Sort by absolute value of metric, take top n
    data = df.copy()
    data['_abs_metric'] = data[metric_col].abs()
    data = data.sort_values('_abs_metric', ascending=False).head(n)

    # Reverse for horizontal bar chart (top item at top)
    data = data.iloc[::-1].reset_index(drop=True)

    n_bars = len(data)
    fig_height = max(min_fig_height, n_bars * figsize_per_row)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Color by direction and significance
    colors = []
    for _, row in data.iterrows():
        if sig_col in data.columns and not row.get(sig_col, False):
            colors.append(COLOR_NS)
        elif row[metric_col] > 0:
            colors.append(COLOR_HIGH)
        else:
            colors.append(COLOR_LOW)

    # Truncate long entity names
    labels = [
        name[:50] + '...' if len(str(name)) > 50 else str(name)
        for name in data[entity_col]
    ]

    y_pos = np.arange(n_bars)
    bars = ax.barh(y_pos, data[metric_col], color=colors, alpha=0.8,
                   edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # Zero line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Annotations
    if annotate_effect_size and effect_col in data.columns:
        for i, (bar, row_idx) in enumerate(zip(bars, data.index)):
            row = data.loc[row_idx]
            r_val = row.get(effect_col, 0)
            x_pos = bar.get_width()

            # Place annotation outside the bar
            ha = 'left' if x_pos >= 0 else 'right'
            offset = abs(data[metric_col].max() - data[metric_col].min()) * 0.02
            x_text = x_pos + offset if x_pos >= 0 else x_pos - offset

            sig_marker = ''
            if sig_col in data.columns and row.get(sig_col, False):
                sig_marker = ' *'

            ax.text(
                x_text, bar.get_y() + bar.get_height() / 2,
                f'r={r_val:.2f}{sig_marker}',
                ha=ha, va='center', fontsize=7, color='#424242'
            )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.8,
                       label=f"Higher in {group_names['low']}"),
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.8,
                       label=f"Higher in {group_names['high']}"),
    ]
    if sig_col in data.columns:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_NS, alpha=0.8,
                           label='Not significant (FDR)')
        )
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    # Labels
    metric_labels = {
        'rank_difference': 'Mean Rank Difference (Low ← | → High)',
        'rank_biserial_r': 'Rank-Biserial Correlation',
    }
    ax.set_xlabel(
        metric_labels.get(metric_col, metric_col.replace('_', ' ').title()),
        fontsize=11
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved bar plot to {output_path}")


def plot_top_differential_bars(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    n: int = 20,
    entity_col: str = 'entity',
    group_names: dict = None,
):
    """
    Simplified bar chart showing top differential entities by effect size.

    Uses rank_biserial_r as the metric (effect size), which is more
    interpretable than raw rank differences for publication.

    Args:
        df: DataFrame from ImportanceAnalyzer.
        title: Plot title.
        output_path: Output path.
        n: Number of entities.
        entity_col: Column with entity names.
        group_names: Display names for groups.
    """
    plot_rank_difference_bars(
        df=df,
        title=title,
        output_path=output_path,
        n=n,
        entity_col=entity_col,
        metric_col='rank_biserial_r',
        annotate_effect_size=False,
        annotate_fdr=True,
        group_names=group_names,
    )


def create_all_bar_plots(
    analysis_dir: str,
    output_dir: str,
    n: int = 30,
    group_names: dict = None,
):
    """
    Auto-generate bar plots from all rank analysis CSVs in a directory.

    Scans for *_rank_analysis.csv and *_top_differential.csv files.

    Args:
        analysis_dir: Directory with analysis CSVs.
        output_dir: Directory for plot output.
        n: Number of entities per plot.
        group_names: Display names for groups.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map analysis names to readable titles
    title_map = {
        'pathway_gate': 'Pathway Gate Importance',
        'gene_average': 'Gene Importance (Average)',
        'gene_sum': 'Gene Importance (Sum)',
        'prototype_raw': 'Prototype Importance (WSI Gate)',
        'prototype_attended': 'Prototype Importance (After Pathway Attention)',
        'prototype_assignment_freq': 'Prototype Assignment Frequency',
    }

    csv_files = sorted(
        f for f in os.listdir(analysis_dir)
        if f.endswith('_rank_analysis.csv')
    )

    for csv_file in csv_files:
        analysis_name = csv_file.replace('_rank_analysis.csv', '')
        df = pd.read_csv(os.path.join(analysis_dir, csv_file))

        if len(df) == 0:
            continue

        # Determine title
        if analysis_name in title_map:
            base_title = title_map[analysis_name]
        elif analysis_name.startswith('crossmodal_proto_'):
            idx = analysis_name.split('_')[-1]
            base_title = f'Prototype {idx}: Pathway Attention'
        elif analysis_name.startswith('genes_in_'):
            pathway = analysis_name.replace('genes_in_', '').replace('_', ' ')
            base_title = f'Gene Attention in {pathway}'
        else:
            base_title = analysis_name.replace('_', ' ').title()

        # Rank difference plot
        plot_rank_difference_bars(
            df=df,
            title=f'{base_title}\n(Rank Difference by Risk Group)',
            output_path=os.path.join(
                output_dir, f'{analysis_name}_rank_diff.pdf'
            ),
            n=n,
            group_names=group_names,
        )

        # Effect size plot
        plot_top_differential_bars(
            df=df,
            title=f'{base_title}\n(Rank-Biserial Correlation)',
            output_path=os.path.join(
                output_dir, f'{analysis_name}_effect_size.pdf'
            ),
            n=min(n, 20),
            group_names=group_names,
        )

    logger.info(
        f"Generated bar plots for {len(csv_files)} analyses in {output_dir}"
    )


def _ensure_pdf(path):
    base, ext = os.path.splitext(path)
    if ext.lower() != '.pdf':
        return base + '.pdf'
    return path