"""
Full visualization suite for ProtoPathway interpretability.

Two-tier analysis:
    Cross-fold (prototype-independent signals):
        - Kaplan-Meier survival curves
        - Pathway gate importance (Signal B) — rank-difference bars + violins
        - Gene importance (Signal C) — rank-difference bars + violins
        - Within-pathway gene drill-down (Signal A columns)

    Per-fold (prototype-dependent signals):
        - Prototype raw importance (Signal E) — bars + violins
        - Prototype attended importance (Signal H) — bars + violins
        - Prototype importance shift (E → H) — paired bars, by-risk, slopegraph
        - Prototype assignment frequency (Signal F) — bars + violins
        - Per-prototype cross-modal attention (Signal G) — rank analysis
        - Cross-modal summary heatmap (prototype × pathway)
        - Cross-modal raw attention heatmaps (low/high/comparison)
        - Top prototype-pathway pairs
        - Cross-modal gene drill-down (genes within top cross-modal pathways)

Prototype indices are NOT consistent across folds (see alignment
diagnostics), so all prototype-level analysis is strictly per-fold.
The user selects which fold(s) to showcase in the paper.
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
    'prototype_assignment_freq', 'crossmodal_proto',
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
    n_violin: int = 15,
    n_pathways_per_direction: int = 5,
    top_k_crossmodal_pathways: int = 20,
    n_crossmodal_gene_drilldown: int = 5,
):
    """
    Visualization pipeline with per-fold prototype analysis.

    Cross-fold steps (prototype-independent):
        1. Pool CV fold data
        2. Kaplan-Meier curves
        3. Pathway and gene importance analysis + bar plots
        3b. Within-pathway gene analysis
        4. Cross-fold bar plots
        4b. Cross-fold violin plots

    Per-fold steps (prototype-dependent):
        5. For each fold independently:
           - Prototype importance (raw + attended) bars + violins
           - Assignment frequency bars + violins
           - Prototype shift (E → H) paired bars, by-risk, slopegraph
           - Per-prototype cross-modal rank analysis
           - Cross-modal summary heatmap
           - Cross-modal raw attention heatmaps
           - Cross-modal gene drill-down (top pathways per risk group)

    Args:
        eval_dir: Path to evaluation directory.
        output_dir: Output directory for figures.
        entity_names: Dict with 'gene_names' and 'pathway_names'.
        risk_stratification: 'median' or 'quartile'.
        n_bar: Number of entities in bar plots.
        n_violin: Number of entities in violin plots.
        n_pathways_per_direction: Pathways per risk direction for gene drill-down.
        top_k_crossmodal_pathways: Pathways to show in cross-modal heatmap.
        n_crossmodal_gene_drilldown: Pathways per risk direction for cross-modal
            gene drill-down.
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
    violin_dir = output_dir / 'violin_plots'
    perfold_dir = output_dir / 'per_fold'

    for d in [analysis_dir, km_dir, bar_dir, violin_dir, perfold_dir]:
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
    try:
        from utils.visualization.diagnose_cross_modal_entropy import (
            diagnose_crossmodal_entropy,
        )
        diagnose_crossmodal_entropy(
            attention_by_patient, predictions, logger=logger
        )
    except ImportError:
        logger.debug("Cross-modal entropy diagnostics not available")

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
    logger.info("Step 4: Cross-fold bar plots (pathway + gene)")
    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ── Step 4b: Cross-fold violin plots ─────────────────────────────
    logger.info("Step 4b: Cross-fold violin plots (pathway + gene)")

    from utils.visualization.gate_violin_plots import (
        plot_gate_importance_violin,
    )

    if 'pathway_gate' in analyzers:
        try:
            plot_gate_importance_violin(
                analyzer=analyzers['pathway_gate'],
                title='Pathway Gate Importance by Risk Group',
                output_path=str(violin_dir / 'pathway_gate_violin.pdf'),
                n=n_violin,
                orientation='horizontal',
            )
        except Exception as e:
            logger.warning(f"Failed to generate pathway gate violin: {e}")

    if 'gene_average' in analyzers:
        try:
            plot_gate_importance_violin(
                analyzer=analyzers['gene_average'],
                title='Gene Importance (Average) by Risk Group',
                output_path=str(violin_dir / 'gene_average_violin.pdf'),
                n=n_violin,
                orientation='horizontal',
            )
        except Exception as e:
            logger.warning(f"Failed to generate gene average violin: {e}")

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
            n_violin=n_violin,
            top_k_crossmodal_pathways=top_k_crossmodal_pathways,
            n_crossmodal_gene_drilldown=n_crossmodal_gene_drilldown,
        )

    logger.info(f"\nVisualization complete. Results in {output_dir}")
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
    n_violin: int = 15,
    top_k_crossmodal_pathways: int = 20,
    n_crossmodal_gene_drilldown: int = 5,
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
            violin_plots/       # Gate importance distributions
            shift_plots/        # E → H shift visualizations
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
    from utils.visualization.gate_violin_plots import (
        plot_gate_importance_violin,
        plot_assignment_frequency_violin,
    )
    from utils.visualization.prototype_shift import (
        plot_prototype_shift,
        plot_shift_slope,
    )

    output_dir = Path(output_dir)
    analysis_dir = output_dir / 'analysis'
    bar_dir = output_dir / 'bar_plots'
    violin_dir = output_dir / 'violin_plots'
    shift_dir = output_dir / 'shift_plots'
    heatmap_dir = output_dir / 'cross_modal'
    gene_drilldown_dir = output_dir / 'crossmodal_gene_drilldown'

    for d in [analysis_dir, bar_dir, violin_dir, shift_dir,
              heatmap_dir, gene_drilldown_dir]:
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
    has_gene_pathway = 'gene_pathway_attention' in sample_attn

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

    # ================================================================
    # Analysis: build all analyzers
    # ================================================================

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

    # ---- Prototype assignment frequency (Signal F) ----
    if has_wsi_gate and 'assignments' in sample_attn.get('patch_assignments', {}):
        logger.info(f"  Prototype assignment frequency (Signal F)...")
        freq_analyzer = FoldStratifiedAnalyzer(
            proto_names, 'prototype_assignment_freq'
        )
        for pid in valid_patients:
            pa = fold_attn[pid]['patch_assignments']
            assignments = np.asarray(pa['assignments']).astype(int)
            n_patches = len(assignments)
            if n_patches == 0:
                continue
            freq = np.zeros(n_protos, dtype=float)
            for proto_i in range(n_protos):
                freq[proto_i] = (assignments == proto_i).sum() / n_patches
            freq_analyzer.add_patient(
                pid, freq, risk_map[pid], fold_idx,
            )
        freq_analyzer.save_results(str(analysis_dir))
        analyzers['prototype_assignment_freq'] = freq_analyzer

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

    # ================================================================
    # Visualization: Bar plots
    # ================================================================
    logger.info(f"  Generating bar plots...")
    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ================================================================
    # Visualization: Violin plots
    # ================================================================
    logger.info(f"  Generating violin plots...")

    if 'prototype_raw' in analyzers:
        try:
            plot_gate_importance_violin(
                analyzer=analyzers['prototype_raw'],
                title=f'Prototype Importance (WSI Gate) by Risk Group\nFold {fold_idx}',
                output_path=str(violin_dir / 'prototype_raw_violin.pdf'),
                n=n_protos,
                orientation='vertical',
            )
        except Exception as e:
            logger.warning(f"  Failed prototype_raw violin: {e}")

    if 'prototype_attended' in analyzers:
        try:
            plot_gate_importance_violin(
                analyzer=analyzers['prototype_attended'],
                title=f'Prototype Importance (After Pathway Attention) by Risk Group\nFold {fold_idx}',
                output_path=str(violin_dir / 'prototype_attended_violin.pdf'),
                n=n_protos,
                orientation='vertical',
            )
        except Exception as e:
            logger.warning(f"  Failed prototype_attended violin: {e}")

    if 'prototype_assignment_freq' in analyzers:
        try:
            plot_assignment_frequency_violin(
                analyzer=analyzers['prototype_assignment_freq'],
                title=f'Prototype Assignment Frequency by Risk Group\nFold {fold_idx}',
                output_path=str(violin_dir / 'assignment_frequency_violin.pdf'),
                n=n_protos,
            )
        except Exception as e:
            logger.warning(f"  Failed assignment frequency violin: {e}")

    # ================================================================
    # Visualization: Prototype shift (E → H)
    # ================================================================
    if 'prototype_shift' in analyzers:
        logger.info(f"  Generating prototype shift plots...")
        shift_analyzer = analyzers['prototype_shift']

        try:
            plot_prototype_shift(
                shift_analyzer=shift_analyzer,
                output_path=str(shift_dir / 'prototype_shift.pdf'),
                title=f'Prototype Importance Shift After Pathway Attention\nFold {fold_idx}',
                show_by_risk=True,
            )
        except Exception as e:
            logger.warning(f"  Failed shift bar plot: {e}")

        try:
            plot_shift_slope(
                shift_analyzer=shift_analyzer,
                output_path=str(shift_dir / 'prototype_shift_slope.pdf'),
                title=f'Prototype Importance: Before vs After Pathway Attention\nFold {fold_idx}',
            )
        except Exception as e:
            logger.warning(f"  Failed shift slope plot: {e}")

    # ================================================================
    # Visualization: Cross-modal summary heatmap
    # ================================================================
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

        try:
            plot_cross_modal_heatmap(
                attention_by_patient=fold_attn_with_risk,
                pathway_names=pathway_names,
                prototype_names=proto_names,
                output_path=str(heatmap_dir / 'cross_modal_average.pdf'),
                title=f'Cross-Modal Attention (Fold {fold_idx} Average)',
            )
        except Exception as e:
            logger.warning(f"  Failed cross-modal average heatmap: {e}")

        try:
            plot_cross_modal_comparison(
                attention_by_patient=fold_attn_with_risk,
                pathway_names=pathway_names,
                prototype_names=proto_names,
                output_dir=str(heatmap_dir),
                top_n_pathways=top_k_crossmodal_pathways,
            )
        except Exception as e:
            logger.warning(f"  Failed cross-modal comparison: {e}")

        try:
            plot_top_prototype_pathway_pairs(
                attention_by_patient=fold_attn_with_risk,
                pathway_names=pathway_names,
                prototype_names=proto_names,
                output_path=str(heatmap_dir / 'top_pairs.pdf'),
                title=f'Top Prototype-Pathway Pairs (Fold {fold_idx})',
            )
        except Exception as e:
            logger.warning(f"  Failed top pairs plot: {e}")

    # ================================================================
    # Cross-modal gene drill-down
    # ================================================================
    if (has_cross_modal and has_gene_pathway
            and pathway_names and gene_names
            and n_crossmodal_gene_drilldown > 0):
        logger.info(
            f"  Cross-modal gene drill-down "
            f"(top {n_crossmodal_gene_drilldown} pathways per risk group)..."
        )
        _run_crossmodal_gene_drilldown(
            analyzers=analyzers,
            fold_attn=fold_attn,
            valid_patients=valid_patients,
            risk_map=risk_map,
            fold_idx=fold_idx,
            pathway_names=pathway_names,
            gene_names=gene_names,
            n_protos=n_protos,
            n_per_direction=n_crossmodal_gene_drilldown,
            analysis_dir=analysis_dir,
            bar_dir=gene_drilldown_dir,
            n_bar=n_bar,
        )

    # Save fold summary
    summary_lines = [
        f"Fold {fold_idx} Prototype Analysis Summary",
        "=" * 50,
        f"Patients: {len(valid_patients)} ({n_low} low risk, {n_high} high risk)",
        f"Prototypes: {n_protos}",
        f"Pathways: {len(pathway_names)}",
        f"Genes: {len(gene_names)}",
        f"Signals analysed: {', '.join(sorted(analyzers.keys()))}",
        "",
        "Output directories:",
        f"  Analysis CSVs: {analysis_dir}",
        f"  Bar plots: {bar_dir}",
        f"  Violin plots: {violin_dir}",
        f"  Shift plots: {shift_dir}",
        f"  Heatmaps: {heatmap_dir}",
        f"  Cross-modal gene drill-down: {gene_drilldown_dir}",
    ]

    summary_path = output_dir / 'fold_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger.info(f"  Fold {fold_idx} complete -> {output_dir}")


# =====================================================================
# Cross-modal gene drill-down
# =====================================================================

def _run_crossmodal_gene_drilldown(
    analyzers: Dict,
    fold_attn: Dict[str, Dict],
    valid_patients: List[str],
    risk_map: Dict[str, str],
    fold_idx: int,
    pathway_names: List[str],
    gene_names: List[str],
    n_protos: int,
    n_per_direction: int,
    analysis_dir,
    bar_dir,
    n_bar: int,
):
    """
    For the pathways most highlighted by cross-modal attention per risk
    group, drill down into which genes within those pathways are driving
    the signal.

    Methodology:
        1. Aggregate cross-modal rank_difference across all prototypes
           per pathway (mean |rank_diff| weighted by prototype importance).
        2. Select top-N pathways most attended in high-risk patients
           (positive aggregate rank_diff) and top-N for low-risk.
        3. For each selected pathway, run within-pathway gene analysis
           using the gene_pathway_attention (Signal A) matrix.

    This is distinct from the cross-fold pathway gate gene drill-down
    (Step 3b) because it identifies pathways that the CROSS-MODAL
    mechanism highlights, not just the pathway gate. A pathway might be
    unimportant by gate weight but highly attended by specific prototypes.
    """
    from pathlib import Path
    from utils.analysis.fold_stratified_analysis import FoldStratifiedAnalyzer

    analysis_dir = Path(analysis_dir)
    bar_dir = Path(bar_dir)

    # Step 1: aggregate cross-modal rank differences across prototypes
    # For each pathway, compute the mean rank_difference across all
    # prototypes that have a cross-modal analysis
    pathway_agg = pd.DataFrame({'pathway': pathway_names})
    pathway_agg['mean_rank_diff'] = 0.0
    n_proto_contributions = 0

    for proto_idx in range(n_protos):
        key = f'crossmodal_proto_{proto_idx}'
        if key not in analyzers:
            continue

        results = analyzers[key].rank_analysis()
        if 'entity' not in results.columns or 'rank_difference' not in results.columns:
            continue

        # Map entity -> rank_difference
        rd_map = dict(zip(results['entity'], results['rank_difference']))
        for i, pw in enumerate(pathway_names):
            pathway_agg.loc[i, 'mean_rank_diff'] += rd_map.get(pw, 0.0)
        n_proto_contributions += 1

    if n_proto_contributions == 0:
        logger.warning("  No cross-modal prototype analyses found for gene drill-down")
        return

    pathway_agg['mean_rank_diff'] /= n_proto_contributions

    # Step 2: select top pathways per direction
    high_risk_pws = (
        pathway_agg[pathway_agg['mean_rank_diff'] > 0]
        .sort_values('mean_rank_diff', ascending=False)
        .head(n_per_direction)['pathway']
        .tolist()
    )
    low_risk_pws = (
        pathway_agg[pathway_agg['mean_rank_diff'] < 0]
        .sort_values('mean_rank_diff', ascending=True)
        .head(n_per_direction)['pathway']
        .tolist()
    )

    selected_pathways = high_risk_pws + low_risk_pws
    logger.info(
        f"  Cross-modal gene drill-down: "
        f"{len(high_risk_pws)} high-risk + {len(low_risk_pws)} low-risk "
        f"pathways selected"
    )

    if not selected_pathways:
        return

    # Save the selection for reference
    selection_df = pathway_agg[
        pathway_agg['pathway'].isin(selected_pathways)
    ].sort_values('mean_rank_diff', key=abs, ascending=False)
    selection_df.to_csv(
        str(analysis_dir / 'crossmodal_gene_drilldown_pathways.csv'),
        index=False,
    )

    # Step 3: within-pathway gene analysis for each selected pathway
    drilldown_analysis_dir = analysis_dir / 'crossmodal_gene_drilldown'
    drilldown_analysis_dir.mkdir(parents=True, exist_ok=True)

    for pw_name in selected_pathways:
        if pw_name not in pathway_names:
            continue
        pw_idx = pathway_names.index(pw_name)

        # Find genes participating in this pathway (non-zero attention)
        gene_mask = np.zeros(len(gene_names), dtype=bool)
        for pid in valid_patients:
            attn = fold_attn[pid]
            if 'gene_pathway_attention' not in attn:
                continue
            col = np.asarray(attn['gene_pathway_attention'])[:, pw_idx]
            gene_mask |= (col > 0)

        participating = [
            gene_names[i] for i in range(len(gene_names)) if gene_mask[i]
        ]
        gene_indices = [
            i for i in range(len(gene_names)) if gene_mask[i]
        ]

        if len(participating) < 2:
            logger.debug(
                f"  Skipping {pw_name}: only {len(participating)} genes"
            )
            continue

        safe_pw = (
            pw_name[:60]
            .replace(' ', '_')
            .replace('/', '_')
            .replace(':', '_')
        )
        aname = f'cm_genes_in_{safe_pw}'

        pw_analyzer = FoldStratifiedAnalyzer(participating, aname)
        for pid in valid_patients:
            attn = fold_attn[pid]
            if 'gene_pathway_attention' not in attn:
                continue
            col = np.asarray(attn['gene_pathway_attention'])[:, pw_idx]
            pw_analyzer.add_patient(
                pid,
                col[gene_indices],
                risk_map[pid],
                fold_idx,
            )
        pw_analyzer.save_results(str(drilldown_analysis_dir))

        # Generate bar plot
        try:
            results = pw_analyzer.rank_analysis()
            if len(results) > 0:
                direction = 'High Risk' if pw_name in high_risk_pws else 'Low Risk'
                plot_rank_bars(
                    df=results,
                    title=f'Gene Attention in {pw_name}\n(Cross-Modal {direction} Pathway, Fold {fold_idx})',
                    output_path=str(bar_dir / f'{aname}.pdf'),
                    n=min(n_bar, len(results)),
                )
        except Exception as e:
            logger.warning(f"  Failed bar plot for {pw_name}: {e}")

    logger.info(
        f"  Cross-modal gene drill-down complete: "
        f"{len(selected_pathways)} pathways analysed"
    )


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
    """
    if group_names is None:
        group_names = {0: 'Low Risk', 1: 'High Risk'}

    df = df.copy()
    if metric_col not in df.columns:
        logger.warning(f"Column '{metric_col}' not in DataFrame, skipping")
        return

    # Sort by absolute rank difference, take top N
    df['abs_rd'] = df[metric_col].abs()
    df = df.sort_values('abs_rd', ascending=False).head(n)
    df = df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    fig_height = max(min_fig_height, len(df) * figsize_per_row)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = [
        COLOR_HIGH if v > 0 else COLOR_LOW
        for v in df[metric_col]
    ]

    y_pos = np.arange(len(df))
    ax.barh(
        y_pos, df[metric_col],
        color=colors, edgecolor='white', linewidth=0.5, alpha=0.85,
    )

    # Labels
    labels = df[entity_col].tolist()
    labels = [l[:55] + '...' if len(str(l)) > 55 else str(l) for l in labels]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel('Rank Difference', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.85,
                       label=f'Higher in {group_names[1]}'),
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.85,
                       label=f'Higher in {group_names[0]}'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    Also skips cross-modal gene drilldown CSVs (handled separately).
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

        # Skip cross-modal gene drilldown (handled separately)
        if analysis_name.startswith('cm_genes_in_'):
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
    """
    csv_files = sorted(
        f for f in os.listdir(analysis_dir)
        if f.startswith('crossmodal_proto_') and f.endswith('_rank_analysis.csv')
    )

    if not csv_files:
        logger.warning("No cross-modal prototype CSVs found for summary heatmap")
        return

    # Build matrix: prototypes × pathways
    all_dfs = {}
    for csv_file in csv_files:
        proto_name = csv_file.replace('_rank_analysis.csv', '')
        proto_idx = int(proto_name.split('_')[-1])
        df = pd.read_csv(os.path.join(analysis_dir, csv_file))
        all_dfs[proto_idx] = dict(zip(df['entity'], df['rank_difference']))

    proto_indices = sorted(all_dfs.keys())
    all_pathways = set()
    for rd_map in all_dfs.values():
        all_pathways.update(rd_map.keys())

    # Select top-K pathways by max absolute rank difference across prototypes
    pathway_max_abs = {}
    for pw in all_pathways:
        max_abs = max(abs(all_dfs[pi].get(pw, 0)) for pi in proto_indices)
        pathway_max_abs[pw] = max_abs

    top_pathways = sorted(
        pathway_max_abs.keys(),
        key=lambda pw: pathway_max_abs[pw],
        reverse=True,
    )[:top_k_pathways]

    # Build matrix
    matrix = np.zeros((len(proto_indices), len(top_pathways)))
    for i, pi in enumerate(proto_indices):
        for j, pw in enumerate(top_pathways):
            matrix[i, j] = all_dfs[pi].get(pw, 0)

    # Truncate pathway names for display
    display_names = [
        pw[:50] + '...' if len(pw) > 50 else pw
        for pw in top_pathways
    ]
    proto_labels = [f'Proto {i}' for i in proto_indices]

    # Plot
    if figsize is None:
        figsize = (max(12, len(top_pathways) * 0.5), max(6, len(proto_indices) * 0.6))
    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(matrix.min()), abs(matrix.max())) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        matrix, cmap='RdBu_r', norm=norm,
        aspect='auto', interpolation='nearest',
    )

    ax.set_xticks(np.arange(len(top_pathways)))
    ax.set_xticklabels(display_names, fontsize=7, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(proto_indices)))
    ax.set_yticklabels(proto_labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Rank Difference\n(red=High Risk, blue=Low Risk)', fontsize=9)

    ax.set_title(
        'Cross-Modal Attention Summary\n(Rank Difference by Risk Group)',
        fontsize=13, fontweight='bold',
    )

    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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