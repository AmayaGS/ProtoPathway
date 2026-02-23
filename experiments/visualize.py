"""
Simplified visualization suite for ProtoPathway interpretability.

Two-tier analysis:
    Cross-fold (prototype-independent signals):
        - Kaplan-Meier survival curves
        - Pathway gate importance (Signal B) — rank-difference bars
        - Gene importance (Signal C) — rank-difference bars
        - Within-pathway gene drill-down

    Per-fold (prototype-dependent signals):
        - Per-prototype cross-modal attention heatmaps (Signal G)
        - Prototype raw importance (Signal E) — bar plots
        - Prototype attended importance (Signal H) — bar plots
        - Prototype importance shift (E → H)
        - Cross-modal summary heatmap (prototype × pathway)

Prototype indices are NOT consistent across folds (see alignment
diagnostics), so all prototype-level analysis is strictly per-fold.
The user selects which fold(s) to showcase in the paper.

Changes from original:
    - Bar plots: colored by direction (blue=low risk, red=high risk),
      rank difference only, no effect size or FDR annotations
    - Cross-modal: per-fold heatmaps instead of cross-fold summary
    - Within-pathway genes: auto-selects top 5 high + top 5 low pathways
    - No violin plots for gate weights
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

# Signals that are safe to pool across folds (no prototype indexing)
CROSSFOLD_SAFE_SIGNALS = {
    'pathway_gate', 'gene_average', 'gene_sum',
}

# Signals that require per-fold analysis (prototype-indexed)
PERFOLD_SIGNALS = {
    'prototype_raw', 'prototype_attended', 'prototype_shift',
    'crossmodal_proto',
}


# =====================================================================
# Main entry point
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
    Visualization pipeline with per-fold prototype analysis.

    Cross-fold steps (prototype-independent):
        1. Pool CV fold data
        2. Kaplan-Meier curves
        3. Pathway and gene importance analysis + bar plots
        4. Within-pathway gene analysis

    Per-fold steps (prototype-dependent):
        5. For each fold independently:
           - Prototype importance (raw + attended) bar plots
           - Prototype shift analysis (E -> H)
           - Per-prototype cross-modal rank analysis
           - Cross-modal summary heatmap
    """
    from pathlib import Path
    from utils.analysis.fold_aggregation import (
        pool_cv_results, save_alignment_report,
    )
    from utils.visualization.km_curves import plot_kaplan_meier_both

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
    perfold_dir = output_dir / 'per_fold'

    for d in [analysis_dir, km_dir, bar_dir, perfold_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Pool data ────────────────────────────────────────────
    logger.info("Step 1: Pooling CV fold data")
    predictions, attention_by_patient, metadata = pool_cv_results(
        str(eval_dir),
        risk_stratification=risk_stratification,
        align_prototypes=False,  # Not needed — per-fold analysis
    )

    # Save alignment diagnostics (for the paper methods section)
    _save_alignment_diagnostics(eval_dir, analysis_dir)

    # Entropy diagnostics
    from utils.visualization.diagnose_cross_modal_entropy import (
        diagnose_crossmodal_entropy,
    )
    diagnose_crossmodal_entropy(
        attention_by_patient, predictions, logger=logger
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
        plot_kaplan_meier_both(
            times=predictions['survival_time'].values,
            events=predictions['event'].values,
            risk_scores=predictions['risk_score'].values,
            output_dir=str(km_dir),
        )

    # ── Step 3: Cross-fold importance analysis ───────────────────────
    # Only pathway and gene signals — no prototype-indexed signals
    logger.info("Step 3: Cross-fold importance analysis (pathway + gene)")

    if use_fold_stratified:
        analyzers = run_fold_stratified_importance_analysis(
            predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=str(analysis_dir),
            pathways_of_interest=[],
            top_k_pathways=0,
            skip_prototype_signals=True,
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

    # Smart pathway selection for gene drill-down
    pathways_of_interest = []
    if 'pathway_gate' in analyzers:
        pathways_of_interest = select_top_pathways_by_direction(
            analyzers['pathway_gate'],
            n_per_direction=n_pathways_per_direction,
        )

    # Within-pathway gene analysis
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
                skip_prototype_signals=True,
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

    # ── Step 4: Cross-fold bar plots ─────────────────────────────────
    logger.info("Step 4: Cross-fold bar plots (pathway + gene only)")
    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ── Step 5: Per-fold prototype analysis ──────────────────────────
    logger.info("Step 5: Per-fold prototype analysis")

    fold_indices = sorted(predictions['fold'].unique())
    for fold_idx in fold_indices:
        logger.info(f"\n{'─'*40}")
        logger.info(f"  Per-fold analysis: Fold {fold_idx}")
        logger.info(f"{'─'*40}")

        _run_per_fold_prototype_analysis(
            fold_idx=int(fold_idx),
            predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=perfold_dir / f'fold_{fold_idx}',
            n_bar=n_bar,
            top_k_crossmodal_pathways=top_k_crossmodal_pathways,
        )

    logger.info(f"\nVisualization complete. Per-fold results in {perfold_dir}")
    return analyzers


# =====================================================================
# Alignment diagnostics (saved but not used for remapping)
# =====================================================================

def _save_alignment_diagnostics(eval_dir, analysis_dir):
    """Run prototype alignment for diagnostic purposes only."""
    from utils.analysis.fold_aggregation import (
        _load_trained_prototypes,
        _compute_prototype_alignment,
        _log_alignment_diagnostics,
        save_alignment_report,
    )

    try:
        prototypes = _load_trained_prototypes(eval_dir.parent)
        if prototypes is None or len(prototypes) < 2:
            return

        permutations, similarities = _compute_prototype_alignment(prototypes)
        _log_alignment_diagnostics(
            permutations, similarities, prototypes[0].shape[0]
        )

        from utils.analysis.fold_aggregation import (
            _compute_per_prototype_stability,
        )
        stability = _compute_per_prototype_stability(
            similarities, prototypes[0].shape[0], permutations
        )

        save_alignment_report(
            {
                'reference_fold': 0,
                'num_folds': len(prototypes),
                'num_prototypes': prototypes[0].shape[0],
                'permutations': permutations,
                'per_fold_similarities': {
                    k: v.tolist() for k, v in similarities.items()
                },
                'per_fold_mean_similarity': {
                    k: float(v.mean()) for k, v in similarities.items()
                },
                'per_prototype_stability': stability,
            },
            str(analysis_dir / 'prototype_alignment_report.txt'),
        )
    except Exception as e:
        logger.warning(f"Could not run alignment diagnostics: {e}")


# =====================================================================
# Per-fold prototype analysis
# =====================================================================

def _run_per_fold_prototype_analysis(
    fold_idx: int,
    predictions: pd.DataFrame,
    attention_by_patient: Dict[str, Dict],
    entity_names: Dict[str, List[str]],
    output_dir,
    n_bar: int = 30,
    top_k_crossmodal_pathways: int = 20,
):
    """
    Run all prototype-dependent analyses for a single fold.

    Within a single fold, prototype indices are coherent (same model),
    so per-prototype cross-modal analysis, prototype importance, and
    prototype shift are all valid.

    Output structure:
        output_dir/
            analysis/           # CSVs from importance analysis
            bar_plots/          # Prototype importance bars
            cross_modal/        # Cross-modal heatmaps
    """
    from pathlib import Path
    from utils.analysis.fold_stratified_analysis import (
        FoldStratifiedAnalyzer,
        PrototypeShiftAnalyzer,
    )
    from utils.visualization.cross_modal_heatmaps import (
        plot_cross_modal_heatmap,
        plot_cross_modal_comparison,
        plot_top_prototype_pathway_pairs,
    )

    output_dir = Path(output_dir)
    analysis_dir = output_dir / 'analysis'
    bar_dir = output_dir / 'bar_plots'
    heatmap_dir = output_dir / 'cross_modal'

    for d in [analysis_dir, bar_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Filter to this fold's patients
    fold_preds = predictions[predictions['fold'] == fold_idx]
    fold_patient_ids = set(fold_preds['patient_id'].values)

    fold_attn = {
        pid: attn for pid, attn in attention_by_patient.items()
        if pid in fold_patient_ids
    }

    risk_map = dict(
        zip(fold_preds['patient_id'], fold_preds['risk_group'])
    )

    valid_patients = [
        pid for pid in fold_attn
        if pid in risk_map and risk_map[pid] is not None
    ]

    n_low = sum(1 for pid in valid_patients if risk_map[pid] == 'Low Risk')
    n_high = sum(1 for pid in valid_patients if risk_map[pid] == 'High Risk')
    logger.info(
        f"  Fold {fold_idx}: {len(valid_patients)} patients "
        f"({n_low} low risk, {n_high} high risk)"
    )

    if len(valid_patients) < 10:
        logger.warning(
            f"  Fold {fold_idx}: too few patients ({len(valid_patients)}), "
            f"skipping prototype analysis"
        )
        return

    # Detect available signals
    sample_attn = fold_attn[valid_patients[0]]
    pathway_names = entity_names.get('pathway_names', [])
    gene_names = entity_names.get('gene_names', [])

    has_wsi_gate = (
        'patch_assignments' in sample_attn
        and isinstance(sample_attn['patch_assignments'], dict)
        and 'gate_weights' in sample_attn['patch_assignments']
    )
    has_fusion_gate = 'fusion_gate_weights' in sample_attn
    has_cross_modal = 'cross_modal_attention' in sample_attn

    n_protos = None
    if has_wsi_gate:
        n_protos = len(sample_attn['patch_assignments']['gate_weights'])
    elif has_cross_modal:
        n_protos = sample_attn['cross_modal_attention'].shape[0]

    if n_protos is None:
        logger.warning(f"  Fold {fold_idx}: no prototype signals found")
        return

    proto_names = [f'Prototype {i}' for i in range(n_protos)]
    analyzers = {}

    # ---- Prototype raw importance (Signal E) ----
    if has_wsi_gate:
        logger.info(f"  Prototype raw importance (Signal E)...")
        analyzer = FoldStratifiedAnalyzer(proto_names, 'prototype_raw')
        for pid in valid_patients:
            attn = fold_attn[pid]
            analyzer.add_patient(
                pid,
                attn['patch_assignments']['gate_weights'],
                risk_map[pid],
                fold_idx,
            )
        analyzer.save_results(str(analysis_dir))
        analyzers['prototype_raw'] = analyzer

    # ---- Prototype attended importance (Signal H) ----
    if has_fusion_gate:
        logger.info(f"  Prototype attended importance (Signal H)...")
        analyzer = FoldStratifiedAnalyzer(proto_names, 'prototype_attended')
        for pid in valid_patients:
            attn = fold_attn[pid]
            analyzer.add_patient(
                pid,
                attn['fusion_gate_weights'],
                risk_map[pid],
                fold_idx,
            )
        analyzer.save_results(str(analysis_dir))
        analyzers['prototype_attended'] = analyzer

    # ---- Prototype shift (E -> H) ----
    if has_wsi_gate and has_fusion_gate:
        logger.info(f"  Prototype shift (E -> H)...")
        shift_analyzer = PrototypeShiftAnalyzer(proto_names)
        for pid in valid_patients:
            attn = fold_attn[pid]
            shift_analyzer.add_patient(
                pid,
                wsi_gate=attn['patch_assignments']['gate_weights'],
                fusion_gate=attn['fusion_gate_weights'],
                risk_group=risk_map[pid],
            )
        shift_analyzer.save_results(str(analysis_dir))
        analyzers['prototype_shift'] = shift_analyzer

    # ---- Per-prototype cross-modal pathway ranking (Signal G) ----
    if has_cross_modal and pathway_names:
        logger.info(
            f"  Per-prototype cross-modal attention "
            f"({n_protos} prototypes x {len(pathway_names)} pathways)..."
        )
        for proto_idx in range(n_protos):
            name = f'crossmodal_proto_{proto_idx}'
            analyzer = FoldStratifiedAnalyzer(pathway_names, name)
            for pid in valid_patients:
                attn = fold_attn[pid]
                analyzer.add_patient(
                    pid,
                    attn['cross_modal_attention'][proto_idx],
                    risk_map[pid],
                    fold_idx,
                )
            analyzer.save_results(str(analysis_dir))
            analyzers[name] = analyzer

    # ---- Bar plots for prototype signals ----
    logger.info(f"  Generating bar plots...")
    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ---- Cross-modal summary heatmap ----
    logger.info(f"  Generating cross-modal summary heatmap...")
    plot_crossmodal_summary_heatmap(
        analysis_dir=str(analysis_dir),
        output_path=str(heatmap_dir / 'crossmodal_summary.pdf'),
        pathway_names=pathway_names,
        top_k_pathways=top_k_crossmodal_pathways,
    )

    # ---- Cross-modal attention heatmaps (raw attention values) ----
    if has_cross_modal and pathway_names:
        logger.info(f"  Generating cross-modal attention heatmaps...")

        # Build fold attention dict with risk_group for heatmap functions
        fold_attn_with_risk = {}
        for pid in valid_patients:
            fold_attn_with_risk[pid] = dict(fold_attn[pid])
            fold_attn_with_risk[pid]['risk_group'] = risk_map[pid]

        plot_cross_modal_heatmap(
            attention_by_patient=fold_attn_with_risk,
            pathway_names=pathway_names,
            prototype_names=proto_names,
            output_path=str(heatmap_dir / 'cross_modal_average.pdf'),
            title=f'Cross-Modal Attention (Fold {fold_idx} Average)',
        )

        plot_cross_modal_comparison(
            attention_by_patient=fold_attn_with_risk,
            pathway_names=pathway_names,
            prototype_names=proto_names,
            output_dir=str(heatmap_dir),
            top_n_pathways=top_k_crossmodal_pathways,
        )

        plot_top_prototype_pathway_pairs(
            attention_by_patient=fold_attn_with_risk,
            pathway_names=pathway_names,
            prototype_names=proto_names,
            output_path=str(heatmap_dir / 'top_pairs.pdf'),
            title=f'Top Prototype-Pathway Pairs (Fold {fold_idx})',
        )

    # Save fold summary
    summary_lines = [
        f"Fold {fold_idx} Prototype Analysis Summary",
        "=" * 50,
        f"Patients: {len(valid_patients)} ({n_low} low risk, {n_high} high risk)",
        f"Prototypes: {n_protos}",
        f"Pathways: {len(pathway_names)}",
        f"Signals analysed: {', '.join(sorted(analyzers.keys()))}",
        "",
        "Output directories:",
        f"  Analysis CSVs: {analysis_dir}",
        f"  Bar plots: {bar_dir}",
        f"  Heatmaps: {heatmap_dir}",
    ]

    summary_path = output_dir / 'fold_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger.info(f"  Fold {fold_idx} complete -> {output_dir}")


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
    if metric_col not in df.columns:
        logger.warning(f"Column '{metric_col}' not in DataFrame, skipping")
        return

    if group_names is None:
        group_names = {'low': 'Low Risk', 'high': 'High Risk'}

    # Select top entities by absolute rank difference
    sort_col = (
        'abs_rank_difference' if 'abs_rank_difference' in df.columns
        else metric_col
    )
    data = df.nlargest(n, sort_col, keep='first').copy()
    data = data.sort_values(metric_col, ascending=True)

    if len(data) == 0:
        return

    fig_height = max(min_fig_height, len(data) * figsize_per_row)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in data[metric_col]]

    entities = data[entity_col].values
    display_names = [
        e[:50] + '...' if len(str(e)) > 50 else str(e) for e in entities
    ]

    ax.barh(
        range(len(data)), data[metric_col].values,
        color=colors, alpha=0.85, edgecolor='white', linewidth=0.5
    )
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(display_names, fontsize=9)

    ax.axvline(0, color='black', linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.85,
                       label=f"Higher in {group_names['low']}"),
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.85,
                       label=f"Higher in {group_names['high']}"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    ax.set_xlabel(
        'Mean Rank Difference (<- Low Risk | High Risk ->)', fontsize=11
    )
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

    When used in per-fold mode, this is coherent because all prototypes
    come from the same trained model.

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
        pathway_to_rdiff = dict(zip(df['entity'], df['rank_difference']))
        for j, pw in enumerate(all_pathways):
            rdiff_matrix[proto_idx, j] = pathway_to_rdiff.get(pw, 0.0)

    # Select top-K pathways by max |rank_difference| across any prototype
    max_abs_rdiff = np.max(np.abs(rdiff_matrix), axis=0)
    top_indices = np.argsort(max_abs_rdiff)[-top_k_pathways:][::-1]

    selected_matrix = rdiff_matrix[:, top_indices]
    selected_names = [all_pathways[i] for i in top_indices]

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
        cmap='RdBu_r',
        norm=norm,
    )

    ax.set_yticks(range(n_protos))
    ax.set_yticklabels([f'Proto {i}' for i in range(n_protos)], fontsize=9)
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=7, rotation=60, ha='right')

    ax.set_title(
        'Cross-Modal Attention: Rank Difference by Prototype x Pathway',
        fontsize=12, fontweight='bold', pad=12,
    )
    ax.set_ylabel('Prototype', fontsize=10)
    ax.set_xlabel('Pathway', fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(
        'Mean Rank Difference\n(<- Low Risk | High Risk ->)',
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

    high_risk_pws = (
        results[results['rank_difference'] > 0]
        .sort_values('rank_difference', ascending=False)
        .head(n_per_direction)['entity']
        .tolist()
    )

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


# ── Utility ──────────────────────────────────────────────────────────

def _ensure_pdf(path):
    base, ext = os.path.splitext(str(path))
    return base + '.pdf' if ext.lower() != '.pdf' else str(path)