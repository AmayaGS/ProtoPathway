"""
Simplified visualization suite for ProtoPathway interpretability.

Changes from original:
    - Bar plots: colored by direction (blue=low risk, red=high risk),
      rank difference only, no effect size or FDR annotations
    - Cross-modal: single summary heatmap instead of per-prototype bar plots
    - Within-pathway genes: auto-selects top 5 high + top 5 low pathways
    - No violin plots for gate weights (near-uniform distributions make
      them uninformative; rank analysis captures the real signal)

Drop-in replacements for functions in utils/visualization/.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)

# ── Palette ──────────────────────────────────────────────────────────
COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'


# =====================================================================
# 1. Simplified rank-difference bar plot
# =====================================================================

def plot_rank_bars(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    n: int = 30,
    entity_col: str = 'entity',
    metric_col: str = 'rank_difference',
    group_names: Optional[Dict] = None,
    figsize_per_row: float = 0.35,
    min_fig_height: float = 6,
):
    """
    Clean horizontal bar chart of rank differences.

    Blue bars = higher importance in Low Risk.
    Red bars  = higher importance in High Risk.
    No effect-size annotations, no FDR markers, no grey bars.

    Args:
        df: DataFrame with at least `entity_col` and `metric_col`.
        title: Plot title.
        output_path: Save path (forced to .pdf).
        n: Max entities to show.
        entity_col: Column with entity names.
        metric_col: Column to plot (default 'rank_difference').
        group_names: Display names for legend.
        figsize_per_row: Height per bar.
        min_fig_height: Minimum figure height.
    """
    group_names = group_names or {'low': 'Low Risk', 'high': 'High Risk'}

    data = df.copy()
    data['_abs'] = data[metric_col].abs()
    data = data.sort_values('_abs', ascending=False).head(n)
    data = data.iloc[::-1].reset_index(drop=True)  # top item at top

    n_bars = len(data)
    fig_height = max(min_fig_height, n_bars * figsize_per_row)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = [
        COLOR_HIGH if v > 0 else COLOR_LOW
        for v in data[metric_col]
    ]

    labels = [
        str(name)[:50] + '...' if len(str(name)) > 50 else str(name)
        for name in data[entity_col]
    ]

    y_pos = np.arange(n_bars)
    ax.barh(y_pos, data[metric_col], color=colors, alpha=0.85,
            edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='black', alpha=0.3, linewidth=1)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.85,
                       label=f"Higher in {group_names['low']}"),
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.85,
                       label=f"Higher in {group_names['high']}"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    ax.set_xlabel('Mean Rank Difference (Low ← | → High)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved bar plot to {output_path}")


def create_all_bar_plots(
    analysis_dir: str,
    output_dir: str,
    n: int = 30,
    group_names: Optional[Dict] = None,
):
    """
    Auto-generate simplified bar plots from all rank analysis CSVs.

    Skips per-prototype cross-modal CSVs (those get the summary heatmap).
    """
    os.makedirs(output_dir, exist_ok=True)

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

        # Skip per-prototype cross-modal (handled by heatmap)
        if analysis_name.startswith('crossmodal_proto_'):
            continue

        df = pd.read_csv(os.path.join(analysis_dir, csv_file))
        if len(df) == 0:
            continue

        if analysis_name in title_map:
            base_title = title_map[analysis_name]
        elif analysis_name.startswith('genes_in_'):
            pathway = analysis_name.replace('genes_in_', '').replace('_', ' ')
            base_title = f'Gene Attention in {pathway}'
        else:
            base_title = analysis_name.replace('_', ' ').title()

        plot_rank_bars(
            df=df,
            title=base_title,
            output_path=os.path.join(output_dir, f'{analysis_name}.pdf'),
            n=n,
            group_names=group_names,
        )

    logger.info(f"Generated bar plots in {output_dir}")


# =====================================================================
# 2. Cross-modal summary heatmap
# =====================================================================

def plot_crossmodal_summary_heatmap(
    analysis_dir: str,
    output_path: str,
    pathway_names: Optional[List[str]] = None,
    top_k_pathways: int = 20,
    figsize: Optional[Tuple] = None,
):
    """
    Single heatmap summarising cross-modal attention across all prototypes.

    Rows = prototypes, columns = top-K most differential pathways
    (selected by max |rank_difference| across any prototype).
    Cell colour = mean rank difference (blue = low risk, red = high risk).

    Args:
        analysis_dir: Directory with crossmodal_proto_*_rank_analysis.csv.
        output_path: Save path.
        pathway_names: Optional full pathway name list for display.
        top_k_pathways: Number of pathways to show.
        figsize: Figure size (auto-calculated if None).
    """
    # Load all per-prototype CSVs
    csv_files = sorted(
        f for f in os.listdir(analysis_dir)
        if f.startswith('crossmodal_proto_') and f.endswith('_rank_analysis.csv')
    )

    if not csv_files:
        logger.warning("No cross-modal CSVs found, skipping heatmap")
        return

    # Build matrix: [n_prototypes, n_pathways]
    proto_dfs = {}
    for csv_file in csv_files:
        name = csv_file.replace('_rank_analysis.csv', '')
        idx = int(name.split('_')[-1])
        df = pd.read_csv(os.path.join(analysis_dir, csv_file))
        proto_dfs[idx] = df

    n_protos = max(proto_dfs.keys()) + 1
    sample_df = next(iter(proto_dfs.values()))
    all_pathways = sample_df['entity'].tolist()
    n_pathways = len(all_pathways)

    # Build rank-difference matrix
    rdiff_matrix = np.zeros((n_protos, n_pathways))
    for proto_idx, df in proto_dfs.items():
        # Ensure consistent pathway ordering
        pathway_to_rdiff = dict(zip(df['entity'], df['rank_difference']))
        for j, pw in enumerate(all_pathways):
            rdiff_matrix[proto_idx, j] = pathway_to_rdiff.get(pw, 0.0)

    # Select top-K pathways by max |rank_difference| across any prototype
    max_abs_rdiff = np.max(np.abs(rdiff_matrix), axis=0)  # [n_pathways]
    top_indices = np.argsort(max_abs_rdiff)[-top_k_pathways:][::-1]

    selected_matrix = rdiff_matrix[:, top_indices]
    selected_names = [all_pathways[i] for i in top_indices]

    # Truncate pathway names for display
    display_names = [
        n[:40] + '...' if len(n) > 40 else n for n in selected_names
    ]

    # Plot
    if figsize is None:
        figsize = (max(10, top_k_pathways * 0.5), max(5, n_protos * 0.4))

    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.max(np.abs(selected_matrix))
    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(
        selected_matrix, aspect='auto',
        cmap='RdBu_r',  # red = high risk, blue = low risk
        norm=norm,
    )

    ax.set_yticks(range(n_protos))
    ax.set_yticklabels([f'Proto {i}' for i in range(n_protos)], fontsize=9)
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=7, rotation=60, ha='right')

    ax.set_title(
        'Cross-Modal Attention: Rank Difference by Prototype × Pathway',
        fontsize=12, fontweight='bold', pad=12,
    )
    ax.set_ylabel('Prototype', fontsize=10)
    ax.set_xlabel('Pathway', fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(
        'Mean Rank Difference\n(← Low Risk | High Risk →)',
        fontsize=9,
    )

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved cross-modal heatmap to {output_path}")


# =====================================================================
# 3. Smart pathway selection for within-pathway gene analysis
# =====================================================================

def select_top_pathways_by_direction(
    pathway_analyzer,
    n_per_direction: int = 5,
) -> List[str]:
    """
    Select top pathways from each risk direction for gene drill-down.

    Args:
        pathway_analyzer: ImportanceAnalyzer or FoldStratifiedAnalyzer
            for pathway gate importance.
        n_per_direction: Number of pathways per direction.

    Returns:
        List of pathway names (up to 2 * n_per_direction).
    """
    results = pathway_analyzer.rank_analysis()

    # Top pathways higher in high risk (positive rank_difference)
    high_risk_pws = (
        results[results['rank_difference'] > 0]
        .sort_values('rank_difference', ascending=False)
        .head(n_per_direction)['entity']
        .tolist()
    )

    # Top pathways higher in low risk (negative rank_difference)
    low_risk_pws = (
        results[results['rank_difference'] < 0]
        .sort_values('rank_difference', ascending=True)
        .head(n_per_direction)['entity']
        .tolist()
    )

    selected = high_risk_pws + low_risk_pws
    logger.info(
        f"Selected {len(high_risk_pws)} high-risk + "
        f"{len(low_risk_pws)} low-risk pathways for gene drill-down"
    )
    return selected


# =====================================================================
# Updated visualize.py orchestration
# =====================================================================

def run_simplified_visualization(
    eval_dir: str,
    output_dir: Optional[str] = None,
    entity_names: Optional[Dict[str, List[str]]] = None,
    risk_stratification: str = 'median',
    n_bar: int = 30,
    n_pathways_per_direction: int = 5,
    top_k_crossmodal_pathways: int = 20,
):
    """
    Simplified visualization pipeline.

    Steps:
        1. Pool CV fold data
        2. Kaplan-Meier curves (unchanged)
        3. Fold-stratified importance analysis
        4. Rank-difference bar plots (simplified)
        5. Cross-modal summary heatmap
        6. Within-pathway gene analysis (top 5 high + top 5 low)

    No violin plots for gate weights.
    No per-prototype cross-modal bar plots.
    No effect-size / FDR annotations on bars.
    """
    from pathlib import Path
    from utils.analysis.fold_aggregation import pool_cv_results
    from utils.visualization.km_curves import plot_kaplan_meier_both

    # Try fold-stratified first, fall back to standard
    try:
        from utils.analysis.fold_stratified_analysis import (
            run_fold_stratified_importance_analysis,
        )
        use_fold_stratified = True
    except ImportError:
        from utils.analysis.importance_analyzer import run_importance_analysis
        use_fold_stratified = False
        logger.warning(
            "fold_stratified_analysis not found, using pooled analysis"
        )

    eval_dir = Path(eval_dir)
    if output_dir is None:
        output_dir = eval_dir.parent / 'figures'
    output_dir = Path(output_dir)

    analysis_dir = output_dir / 'analysis'
    km_dir = output_dir / 'km_curves'
    bar_dir = output_dir / 'bar_plots'
    heatmap_dir = output_dir / 'cross_modal'

    for d in [analysis_dir, km_dir, bar_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Pool data ────────────────────────────────────────────
    logger.info("Step 1: Pooling CV fold data")
    predictions, attention_by_patient, metadata = pool_cv_results(
        str(eval_dir), risk_stratification=risk_stratification
    )

    if entity_names is None:
        entity_names = {}
    if not entity_names.get('gene_names') and 'gene_names' in metadata:
        entity_names['gene_names'] = metadata['gene_names']
    if not entity_names.get('pathway_names') and 'pathway_names' in metadata:
        entity_names['pathway_names'] = metadata['pathway_names']

    # ── Step 2: KM curves ────────────────────────────────────────────
    logger.info("Step 2: Kaplan-Meier curves")
    if 'risk_score' in predictions.columns:
        km_results = plot_kaplan_meier_both(
            times=predictions['survival_time'].values,
            events=predictions['event'].values,
            risk_scores=predictions['risk_score'].values,
            output_dir=str(km_dir),
        )

    # ── Step 3: Importance analysis ──────────────────────────────────
    logger.info("Step 3: Importance analysis")

    # First pass: run without within-pathway genes
    if use_fold_stratified:
        analyzers = run_fold_stratified_importance_analysis(
            predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=str(analysis_dir),
            pathways_of_interest=[],  # skip for now
            top_k_pathways=0,
        )
    else:
        analyzers = run_importance_analysis(
            predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=str(analysis_dir),
            pathways_of_interest=[],
            top_k_pathways=0,
        )

    # Smart pathway selection: top N from each direction
    pathways_of_interest = []
    if 'pathway_gate' in analyzers:
        pathways_of_interest = select_top_pathways_by_direction(
            analyzers['pathway_gate'],
            n_per_direction=n_pathways_per_direction,
        )

    # Second pass: within-pathway gene analysis for selected pathways
    if pathways_of_interest:
        logger.info(
            f"Step 3b: Within-pathway gene analysis "
            f"({len(pathways_of_interest)} pathways)"
        )
        if use_fold_stratified:
            gene_analyzers = run_fold_stratified_importance_analysis(
                predictions=predictions,
                attention_by_patient=attention_by_patient,
                entity_names=entity_names,
                output_dir=str(analysis_dir),
                pathways_of_interest=pathways_of_interest,
                top_k_pathways=0,
            )
        else:
            gene_analyzers = run_importance_analysis(
                predictions=predictions,
                attention_by_patient=attention_by_patient,
                entity_names=entity_names,
                output_dir=str(analysis_dir),
                pathways_of_interest=pathways_of_interest,
                top_k_pathways=0,
            )
        analyzers.update(gene_analyzers)

    # ── Step 4: Bar plots ────────────────────────────────────────────
    logger.info("Step 4: Rank-difference bar plots")
    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ── Step 5: Cross-modal heatmap ──────────────────────────────────
    logger.info("Step 5: Cross-modal summary heatmap")
    plot_crossmodal_summary_heatmap(
        analysis_dir=str(analysis_dir),
        output_path=str(heatmap_dir / 'crossmodal_summary.pdf'),
        pathway_names=entity_names.get('pathway_names'),
        top_k_pathways=top_k_crossmodal_pathways,
    )

    logger.info("Visualization complete")
    return analyzers


# ── Utility ──────────────────────────────────────────────────────────

def _ensure_pdf(path):
    base, ext = os.path.splitext(str(path))
    return base + '.pdf' if ext.lower() != '.pdf' else str(path)