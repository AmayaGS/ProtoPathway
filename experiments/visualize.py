"""
Unified visualization suite for ProtoPathway interpretability.

Single entry point for ALL analysis and figure generation.

Steps:
    1. Pool CV fold data
    2. Kaplan-Meier survival curves
    3. Cross-fold importance (pathway + gene) + within-pathway drill-down
    4. Cross-fold bar + violin plots
    5. Per-fold prototype analysis (bars, violins, shift, cross-modal,
       cross-modal gene drill-down)
    6. Spatial overlays + prototype panels (when wsi_features_dir given)

Usage:
    python main.py visualize --eval_dir output/BLCA/exp/evaluation
    python main.py visualize --eval_dir output/BLCA/exp/evaluation \
        --wsi-features-dir processed/BLCA/wsi_features_per_patient \
        --wsi-dir /path/to/slides --fold 1
    python main.py visualize --eval_dir output/BLCA/exp/evaluation \
        --wsi-features-dir processed/BLCA/wsi_features_per_patient \
        --patient TCGA-FD-A3B4 --fold 1
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

from utils.io import save_figure

logger = logging.getLogger(__name__)

COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'




# =====================================================================
# Main entry point
# =====================================================================

def run_simplified_visualization(eval_dir: str,
                                 output_dir: Optional[str] = None,
                                 entity_names: Optional[Dict[str, List[str]]] = None,
                                 risk_stratification: str = 'median',
                                 n_bar: int = 30,
                                 n_violin: int = 15,
                                 n_pathways_per_direction: int = 5,
                                 top_k_crossmodal_pathways: int = 20,
                                 n_crossmodal_gene_drilldown: int = 5,
                                 wsi_features_dir: Optional[str] = None,
                                 wsi_dir: Optional[str] = None,
                                 spatial_fold: Optional[int] = None,
                                 spatial_patient: Optional[str] = None,
                                 spatial_n_per_group: int = 2,
                                 spatial_downsample: int = 4,
                                 spatial_patch_size: int = 256,
                                 spatial_single_pathway: Optional[str] = None,
                                 spatial_single_gene: Optional[str] =None,
                                 force_recalculate: bool = False):
    from pathlib import Path
    from utils.analysis.fold_aggregation import pool_cv_results, save_alignment_report
    from utils.visualization.km_curves import plot_kaplan_meier_both

    try:
        from utils.analysis.fold_stratified_analysis import run_fold_stratified_importance_analysis
        use_fold_stratified = True
    except ImportError:
        from utils.analysis.importance_analyzer import run_importance_analysis
        use_fold_stratified = False

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

    # ── Step 1 ───────────────────────────────────────────────────────
    logger.info("Step 1: Pooling CV fold data")
    predictions, attention_by_patient, metadata = pool_cv_results(
        str(eval_dir), risk_stratification=risk_stratification,
        align_prototypes=False,
    )
    _save_alignment_diagnostics(eval_dir, analysis_dir)

    try:
        from utils.visualization.diagnose_cross_modal_entropy import diagnose_crossmodal_entropy
        diagnose_crossmodal_entropy(attention_by_patient, predictions, logger=logger)
    except ImportError:
        pass

    if entity_names is None:
        entity_names = {}
    if not entity_names.get('gene_names') and 'gene_names' in metadata:
        entity_names['gene_names'] = metadata['gene_names']
    if not entity_names.get('pathway_names') and 'pathway_names' in metadata:
        entity_names['pathway_names'] = metadata['pathway_names']

    dataset_name = None
    for part in eval_dir.parts:
        if part.startswith('TCGA-'):
            dataset_name = part
            break
    if dataset_name is None:
        dataset_name = eval_dir.parent.parent.name

    # ── Step 2 ───────────────────────────────────────────────────────
    logger.info("Step 2: Kaplan-Meier curves")
    if 'risk_score' in predictions.columns:
        plot_kaplan_meier_both(
            times=predictions['survival_time'].values,
            events=predictions['event'].values,
            risk_scores=predictions['risk_score'].values,
            output_dir=str(km_dir),
            dataset_name=dataset_name
        )

        # Risk distribution by event status (all folds pooled)
        plot_risk_distribution_by_event(
            predictions, fold_idx='all',
            output_path=str(km_dir / 'risk_distribution_by_event.pdf'),
        )

        # Per-fold risk distributions
        for fi in sorted(predictions['fold'].unique()):
            fold_df = predictions[predictions['fold'] == fi]
            plot_risk_distribution_by_event(
                fold_df, fold_idx=int(fi),
                output_path=str(km_dir / f'risk_distribution_fold_{int(fi)}.pdf'),
            )

    # ── Step 3 ───────────────────────────────────────────────────────
    logger.info("Step 3: Cross-fold importance analysis")

    _sample_attn = attention_by_patient[next(iter(attention_by_patient))]
    _expected = []
    if 'pathway_importance' in _sample_attn and entity_names.get('pathway_names'):
        _expected.append('pathway_gate')
    if ('gene_pathway_attention' in _sample_attn
            and 'pathway_importance' in _sample_attn
            and entity_names.get('gene_names')):
        _expected.extend(['gene_average', 'gene_sum'])

    if (not force_recalculate
            and _expected
            and _analysis_csvs_exist(analysis_dir, _expected)):
        logger.info("  CSVs exist — loading from disk "
                     "(use --force-recalculate to rerun)")
        from utils.analysis.fold_stratified_analysis import load_saved_analyzers
        analyzers = load_saved_analyzers(str(analysis_dir))
    else:
        if use_fold_stratified:
            analyzers = run_fold_stratified_importance_analysis(
                predictions=predictions,
                attention_by_patient=attention_by_patient,
                entity_names=entity_names,
                output_dir=str(analysis_dir),
                pathways_of_interest=[], top_k_pathways=0,
                skip_prototype_signals=True,
            )
        else:
            analyzers = run_importance_analysis(
                predictions=predictions,
                attention_by_patient=attention_by_patient,
                entity_names=entity_names,
                output_dir=str(analysis_dir),
                pathways_of_interest=[], top_k_pathways=0,
            )

    pathways_of_interest = []
    if 'pathway_gate' in analyzers:
        pathways_of_interest = _select_top_pathways(
            analyzers['pathway_gate'], n_pathways_per_direction)

    if pathways_of_interest:
        _gene_csv_names = []
        for pw in pathways_of_interest:
            safe = pw[:60].replace(' ', '_').replace('/', '_').replace(':', '_')
            _gene_csv_names.append(f'genes_in_{safe}')

        if (not force_recalculate
                and _analysis_csvs_exist(analysis_dir, _gene_csv_names)):
            logger.info("  Gene drill-down CSVs exist — loading from disk")
            from utils.analysis.fold_stratified_analysis import load_saved_analyzers
            gene_a = load_saved_analyzers(str(analysis_dir), _gene_csv_names)
        else:
            logger.info(f"Step 3b: Gene drill-down "
                         f"({len(pathways_of_interest)} pathways)")
            if use_fold_stratified:
                gene_a = run_fold_stratified_importance_analysis(
                    predictions=predictions,
                    attention_by_patient=attention_by_patient,
                    entity_names=entity_names,
                    output_dir=str(analysis_dir),
                    pathways_of_interest=pathways_of_interest,
                    top_k_pathways=0,
                    skip_prototype_signals=True,
                )
            else:
                gene_a = run_importance_analysis(
                    predictions=predictions,
                    attention_by_patient=attention_by_patient,
                    entity_names=entity_names,
                    output_dir=str(analysis_dir),
                    pathways_of_interest=pathways_of_interest,
                    top_k_pathways=0,
                )
        analyzers.update(gene_a)

    # ── Step 4 ───────────────────────────────────────────────────────
    logger.info("Step 4: Cross-fold bar plots")
    create_all_bar_plots(analysis_dir=str(analysis_dir), output_dir=str(bar_dir), n=n_bar)

    logger.info("Step 4b: Cross-fold violin plots")
    from utils.visualization.gate_violin_plots import plot_gate_importance_violin
    for sig, title in [
        ('pathway_gate', 'Pathway Gate Importance by Risk Group'),
        ('gene_average', 'Gene Importance (Average) by Risk Group'),
    ]:
        if sig in analyzers:
            try:
                plot_gate_importance_violin(
                    analyzer=analyzers[sig], title=title,
                    output_path=str(violin_dir / f'{sig}_violin.pdf'),
                    n=n_violin, orientation='horizontal',
                )
            except Exception as e:
                logger.warning(f"  {sig} violin failed: {e}")

    # ── Step 5 ───────────────────────────────────────────────────────
    logger.info("Step 5: Per-fold prototype analysis")
    fold_indices = sorted(predictions['fold'].unique())
    for fi in fold_indices:
        logger.info(f"\n{'─'*40}\n  Per-fold: Fold {fi}\n{'─'*40}")
        _run_per_fold(
            fold_idx=int(fi), predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=perfold_dir / f'fold_{fi}',
            n_bar=n_bar, n_violin=n_violin,
            top_k_crossmodal_pathways=top_k_crossmodal_pathways,
            n_crossmodal_gene_drilldown=n_crossmodal_gene_drilldown,
            force_recalculate=force_recalculate
        )

    # ── Step 6 ───────────────────────────────────────────────────────
    if wsi_features_dir:
        logger.info("\n" + "=" * 60)
        logger.info("Step 6: Spatial visualization + prototype panels")
        logger.info("=" * 60)
        _run_spatial(
            predictions=predictions,
            attention_by_patient=attention_by_patient,
            entity_names=entity_names,
            output_dir=output_dir, perfold_dir=perfold_dir,
            wsi_features_dir=wsi_features_dir, wsi_dir=wsi_dir,
            fold_idx=spatial_fold if spatial_fold is not None else int(fold_indices[0]),
            patient_id=spatial_patient,
            n_per_group=spatial_n_per_group,
            downsample=spatial_downsample, patch_size=spatial_patch_size,
            single_pathway_name=spatial_single_pathway,
            single_gene_name=spatial_single_gene
        )
    else:
        logger.info("\nStep 6 SKIPPED (no --wsi-features-dir)")

    logger.info(f"\nDone. All outputs in {output_dir}")
    return analyzers


# =====================================================================
# Step 6: Spatial
# =====================================================================

def _run_spatial(
    predictions, attention_by_patient, entity_names, output_dir,
    perfold_dir, wsi_features_dir, wsi_dir, fold_idx,
    patient_id, n_per_group, downsample, patch_size,
    single_pathway_name,
    single_gene_name
):
    from pathlib import Path
    from utils.visualization.spatial_heatmaps import generate_patient_spatial_viz
    from utils.visualization.prototype_panels import (
        plot_cohort_prototype_exemplars,
    )

    output_dir = Path(output_dir)
    spatial_dir = output_dir / 'spatial'
    spatial_dir.mkdir(parents=True, exist_ok=True)

    fold_analysis_dir = perfold_dir / f'fold_{fold_idx}' / 'analysis'
    pathway_names = entity_names.get('pathway_names', [])
    gene_names = entity_names.get('gene_names', [])

    if not fold_analysis_dir.exists():
        logger.error(f"  Per-fold analysis not found: {fold_analysis_dir}")
        return

    # Cohort prototype panels
    logger.info("\n  Cohort prototype panels...")
    proto_dir = spatial_dir / 'prototype_panels'
    try:
        plot_cohort_prototype_exemplars(
            attention_by_patient=attention_by_patient,
            wsi_features_dir=wsi_features_dir,
            output_dir=str(proto_dir),
            top_k_protos=5, n_patches_per_proto=8,
            wsi_dir=wsi_dir, downsample=downsample, patch_size=patch_size,
        )
    except Exception as e:
        logger.error(f"  Cohort exemplars failed: {e}")

    # Select patients
    if patient_id:
        pids = [p.strip() for p in patient_id.split(',')]
    else:
        pids = _auto_select_patients(
            predictions, attention_by_patient, wsi_features_dir, n_per_group, fold_idx)

    if not pids:
        logger.warning("  No valid patients for spatial viz")
        return

    risk_map = dict(zip(predictions['patient_id'], predictions['risk_group']))

    # Per-patient overlays
    for pid in pids:
        if pid not in attention_by_patient:
            logger.warning(f"  {pid}: no attention data")
            continue
        if not _verify_pt(pid, wsi_features_dir):
            continue

        rg = risk_map.get(pid, 'Unknown')
        logger.info(f"\n  Spatial: {pid} ({rg})")
        try:
            generate_patient_spatial_viz(
                patient_id=pid,
                attention_data=attention_by_patient[pid],
                pathway_names=pathway_names,
                gene_names=gene_names,
                fold_analysis_dir=str(fold_analysis_dir),
                output_dir=str(spatial_dir / pid),
                risk_group=rg,
                wsi_features_dir=wsi_features_dir,
                wsi_dir=wsi_dir,
                patch_size=patch_size, downsample=downsample,
                single_pathway_name=single_pathway_name,
                single_gene_name=single_gene_name,
                rank_transform=True,
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")
    #
    # # Cohort prototype panels
    # logger.info("\n  Cohort prototype panels...")
    # proto_dir = spatial_dir / 'prototype_panels'
    # # try:
    # #     plot_prototype_importance(
    # #         attention_by_patient=attention_by_patient,
    # #         output_dir=str(proto_dir), top_k=5, dpi=300,
    # #     )
    # # except Exception as e:
    # #     logger.error(f"  Prototype importance failed: {e}")
    #
    # try:
    #     plot_cohort_prototype_exemplars(
    #         attention_by_patient=attention_by_patient,
    #         wsi_features_dir=wsi_features_dir,
    #         output_dir=str(proto_dir),
    #         top_k_protos=16, n_patches_per_proto=8,
    #         wsi_dir=wsi_dir, downsample=downsample, patch_size=patch_size,
    #     )
    # except Exception as e:
    #     logger.error(f"  Cohort exemplars failed: {e}")

    logger.info(f"  Spatial outputs in {spatial_dir}")


def _auto_select_patients(predictions, attention_by_patient, wsi_features_dir, n, fold_idx):
    import torch

    fold_predictions = predictions[predictions['fold'] == fold_idx]
    logger.info(f"  Auto-selecting from fold {fold_idx} validation set "
                f"({len(fold_predictions)} patients)")
    available = fold_predictions[
        fold_predictions['patient_id'].isin(attention_by_patient.keys())
    ].sort_values('risk_score')

    valid = []
    for pid in available['patient_id']:
        pt = os.path.join(wsi_features_dir, f'{pid}.pt')
        if os.path.exists(pt):
            try:
                d = torch.load(pt, weights_only=False)
                if isinstance(d, dict) and 'coords' in d:
                    valid.append(pid)
            except Exception:
                pass

    vdf = available[available['patient_id'].isin(valid)]
    if len(vdf) == 0:
        return []
    low = vdf.head(n)['patient_id'].tolist()
    high = vdf.tail(n)['patient_id'].tolist()
    sel = low + [p for p in high if p not in low]
    for pid in sel:
        row = available[available['patient_id'] == pid].iloc[0]
        logger.info(f"    {pid}: {row['risk_group']} (score={row['risk_score']:.4f})")
    return sel


def _verify_pt(pid, wsi_features_dir):
    import torch
    pt = os.path.join(wsi_features_dir, f'{pid}.pt')
    if not os.path.exists(pt):
        logger.warning(f"  {pid}: no .pt file")
        return False
    try:
        d = torch.load(pt, weights_only=False)
        if isinstance(d, dict) and 'coords' in d:
            return True
        logger.warning(f"  {pid}: .pt missing coords")
        return False
    except Exception as e:
        logger.warning(f"  {pid}: .pt load error: {e}")
        return False


# =====================================================================
# Alignment diagnostics
# =====================================================================

def _save_alignment_diagnostics(eval_dir, analysis_dir):
    from utils.analysis.fold_aggregation import (
        _load_trained_prototypes, _compute_prototype_alignment,
        _log_alignment_diagnostics, save_alignment_report,
        _compute_per_prototype_stability,
    )
    try:
        prototypes = _load_trained_prototypes(eval_dir.parent)
        if prototypes is None or len(prototypes) < 2:
            return
        perms, sims = _compute_prototype_alignment(prototypes)
        _log_alignment_diagnostics(perms, sims, prototypes[0].shape[0])
        stability = _compute_per_prototype_stability(
            sims, prototypes[0].shape[0], perms)
        save_alignment_report({
            'reference_fold': 0, 'num_folds': len(prototypes),
            'num_prototypes': prototypes[0].shape[0],
            'permutations': perms,
            'per_fold_similarities': {k: v.tolist() for k, v in sims.items()},
            'per_fold_mean_similarity': {k: float(v.mean()) for k, v in sims.items()},
            'per_prototype_stability': stability,
        }, str(analysis_dir / 'prototype_alignment_report.txt'))
    except Exception as e:
        logger.warning(f"Alignment diagnostics failed: {e}")


# =====================================================================
# Per-fold prototype analysis
# =====================================================================

def _run_per_fold(
    fold_idx, predictions, attention_by_patient, entity_names,
    output_dir, n_bar=30, n_violin=15,
    top_k_crossmodal_pathways=20, n_crossmodal_gene_drilldown=5,
    force_recalculate=False
):
    from pathlib import Path
    from utils.analysis.fold_stratified_analysis import (
        FoldStratifiedAnalyzer, PrototypeShiftAnalyzer,
    )
    from utils.visualization.cross_modal_heatmaps import (
        plot_cross_modal_heatmap, plot_cross_modal_comparison,
        plot_top_prototype_pathway_pairs,
    )
    from utils.visualization.gate_violin_plots import (
        plot_gate_importance_violin, plot_assignment_frequency_violin
    )

    output_dir = Path(output_dir)
    dirs = {}
    for name in ['analysis', 'bar_plots', 'violin_plots', 'shift_plots',
                  'cross_modal', 'crossmodal_gene_drilldown']:
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d

    fold_preds = predictions[predictions['fold'] == fold_idx]
    fold_ids = set(fold_preds['patient_id'].values)
    fold_attn = {p: a for p, a in attention_by_patient.items() if p in fold_ids}
    risk_map = dict(zip(fold_preds['patient_id'], fold_preds['risk_group']))
    valid = [p for p in fold_attn if p in risk_map and risk_map[p] is not None]

    n_low = sum(1 for p in valid if risk_map[p] == 'Low Risk')
    n_high = sum(1 for p in valid if risk_map[p] == 'High Risk')
    logger.info(f"  Fold {fold_idx}: {len(valid)} patients ({n_low} low, {n_high} high)")

    if len(valid) < 10:
        logger.warning(f"  Too few patients, skipping"); return

    sample = fold_attn[valid[0]]
    pw_names = entity_names.get('pathway_names', [])
    g_names = entity_names.get('gene_names', [])

    has_e = ('patch_assignments' in sample and isinstance(sample['patch_assignments'], dict)
             and 'gate_weights' in sample['patch_assignments'])
    has_h = 'fusion_gate_weights' in sample
    has_g = 'cross_modal_attention' in sample
    has_a = 'gene_pathway_attention' in sample

    K = None
    if has_e: K = len(sample['patch_assignments']['gate_weights'])
    elif has_g: K = sample['cross_modal_attention'].shape[0]
    if K is None:
        logger.warning(f"  No prototype signals"); return

    pnames = [f'Prototype {i}' for i in range(K)]
    analyzers = {}

    # ── Build analyzers ──────────────────────────────────────────────
    def _build(key, names, extractor):
        a = FoldStratifiedAnalyzer(names, key)
        for p in valid:
            a.add_patient(p, extractor(fold_attn[p]), risk_map[p], fold_idx)
        a.save_results(str(dirs['analysis']))
        analyzers[key] = a

    # Check if per-fold CSVs already exist
    _expected_pf = []
    if has_e: _expected_pf.append('prototype_raw')
    if has_h: _expected_pf.append('prototype_attended')

    if (not force_recalculate
            and _expected_pf
            and _analysis_csvs_exist(dirs['analysis'], _expected_pf)):
        logger.info(f"  Per-fold CSVs exist — loading from disk")
        from utils.analysis.fold_stratified_analysis import load_saved_analyzers
        analyzers = load_saved_analyzers(str(dirs['analysis']))
    else:
        if has_e:
            logger.info(f"  Signal E (WSI gate)...")
            _build('prototype_raw', pnames,
                   lambda a: a['patch_assignments']['gate_weights'])

        if has_h:
            logger.info(f"  Signal H (fusion gate)...")
            _build('prototype_attended', pnames,
                   lambda a: a['fusion_gate_weights'])

        if has_e and has_h:
            logger.info(f"  Shift E->H...")
            sa = PrototypeShiftAnalyzer(pnames)
            for p in valid:
                at = fold_attn[p]
                sa.add_patient(
                    p,
                    wsi_gate=at['patch_assignments']['gate_weights'],
                    fusion_gate=at['fusion_gate_weights'],
                    risk_group=risk_map[p],
                )
            sa.save_results(str(dirs['analysis']))
            analyzers['prototype_shift'] = sa

        # FIX: check 'hard_assignments' first, fall back to 'assignments'
        _pa_sample = sample.get('patch_assignments', {})
        _has_assigns = (
                'hard_assignments' in _pa_sample or 'assignments' in _pa_sample
        )
        if has_e and _has_assigns:
            logger.info(f"  Signal F (assignment freq)...")
            fa = FoldStratifiedAnalyzer(pnames, 'prototype_assignment_freq')
            for p in valid:
                pa = fold_attn[p]['patch_assignments']
                assigns = np.asarray(
                    pa.get('hard_assignments', pa.get('assignments'))
                ).astype(int)
                if len(assigns) == 0:
                    continue
                freq = np.array([
                    (assigns == i).sum() / len(assigns) for i in range(K)
                ])
                fa.add_patient(p, freq, risk_map[p], fold_idx)
            fa.save_results(str(dirs['analysis']))
            analyzers['prototype_assignment_freq'] = fa

        if has_g and pw_names:
            logger.info(f"  Signal G (cross-modal, {K}x{len(pw_names)})...")
            for ki in range(K):
                nm = f'crossmodal_proto_{ki}'
                a = FoldStratifiedAnalyzer(pw_names, nm)
                for p in valid:
                    a.add_patient(
                        p, fold_attn[p]['cross_modal_attention'][ki],
                        risk_map[p], fold_idx,
                    )
                a.save_results(str(dirs['analysis']))
                analyzers[nm] = a

    # ── Risk distribution by event ───────────────────────────────────
    logger.info(f"  Risk distribution by event status...")
    try:
        plot_risk_distribution_by_event(
            fold_preds, fold_idx,
            str(output_dir / f'risk_distribution_fold_{fold_idx}.pdf'),
        )
    except Exception as e:
        logger.warning(f"  Risk distribution: {e}")

    # ── Gating comparison (WSI vs Fusion + delta) ────────────────────
    if has_e and has_h:
        logger.info(f"  Gating rank comparison (WSI vs Fusion)...")
        try:
            plot_gating_comparison(
                fold_attn, risk_map, valid, K, fold_idx,
                str(dirs['shift_plots'] / f'rank_shift_fold_{fold_idx}.pdf'),
            )
        except Exception as e:
            logger.warning(f"  Gating comparison: {e}")

    # ── Bar plots ────────────────────────────────────────────────────
    logger.info(f"  Bar plots...")
    create_all_bar_plots(str(dirs['analysis']), str(dirs['bar_plots']), n_bar)

    # ── Violin plots ─────────────────────────────────────────────────
    logger.info(f"  Violin plots...")
    for key, title, n_e, ori in [
        ('prototype_raw', f'WSI Gate — Fold {fold_idx}', K, 'vertical'),
        ('prototype_attended', f'Fusion Gate — Fold {fold_idx}', K, 'vertical')
    ]:
        if key in analyzers:
            try:
                plot_gate_importance_violin(
                    analyzer=analyzers[key], title=title,
                    output_path=str(dirs['violin_plots'] / f'{key}_violin.pdf'),
                    n=n_e, orientation=ori,
                    sort_by='rank_difference')
            except Exception as e:
                logger.warning(f"  {key} violin: {e}")

        if 'prototype_assignment_freq' in analyzers:
            try:
                plot_assignment_frequency_violin(
                    analyzer=analyzers['prototype_assignment_freq'],
                    title=f'Assignment Frequency — Fold {fold_idx}',
                    output_path=str(dirs['violin_plots'] / 'assignment_freq_violin.pdf')
                )
            except Exception as e:
                logger.warning(f"  Assignment freq violin: {e}")


    # ── Cross-modal heatmaps ─────────────────────────────────────────
    logger.info(f"  Cross-modal heatmaps...")
    plot_crossmodal_summary_heatmap(
        str(dirs['analysis']), str(dirs['cross_modal'] / 'summary.pdf'),
        pw_names, top_k_crossmodal_pathways)

    if has_g and pw_names:
        fa_risk = {}
        for p in valid:
            fa_risk[p] = dict(fold_attn[p])
            fa_risk[p]['risk_group'] = risk_map[p]

        for fn, kw in [
            (plot_cross_modal_heatmap, dict(
                attention_by_patient=fa_risk, pathway_names=pw_names,
                prototype_names=pnames,
                output_path=str(dirs['cross_modal'] / 'average.pdf'),
                title=f'Cross-Modal (Fold {fold_idx})')),
            (plot_cross_modal_comparison, dict(
                attention_by_patient=fa_risk, pathway_names=pw_names,
                prototype_names=pnames, output_dir=str(dirs['cross_modal']),
                top_n_pathways=top_k_crossmodal_pathways)),
            (plot_top_prototype_pathway_pairs, dict(
                attention_by_patient=fa_risk, pathway_names=pw_names,
                prototype_names=pnames,
                output_path=str(dirs['cross_modal'] / 'top_pairs.pdf'),
                title=f'Top Pairs (Fold {fold_idx})')),
        ]:
            try: fn(**kw)
            except Exception as e:
                logger.warning(f"  {fn.__name__}: {e}")

    # ── Cross-modal gene drill-down ──────────────────────────────────
    if has_g and has_a and pw_names and g_names and n_crossmodal_gene_drilldown > 0:
        logger.info(f"  CM gene drill-down...")
        _cm_gene_drilldown(
            analyzers, fold_attn, valid, risk_map, fold_idx,
            pw_names, g_names, K, n_crossmodal_gene_drilldown,
            dirs['analysis'], dirs['crossmodal_gene_drilldown'], n_bar)

    (output_dir / 'fold_summary.txt').write_text(
        f"Fold {fold_idx}: {len(valid)} pts ({n_low}L/{n_high}H), "
        f"{K} protos, {len(pw_names)} pws, {len(g_names)} genes\n"
        f"Signals: {', '.join(sorted(analyzers.keys()))}")
    logger.info(f"  Fold {fold_idx} done → {output_dir}")


# =====================================================================
# Cross-modal gene drill-down
# =====================================================================

def _cm_gene_drilldown(
    analyzers, fold_attn, valid, risk_map, fold_idx,
    pw_names, g_names, K, n_dir, analysis_dir, bar_dir, n_bar,
):
    from pathlib import Path
    from utils.analysis.fold_stratified_analysis import FoldStratifiedAnalyzer

    analysis_dir = Path(analysis_dir)
    bar_dir = Path(bar_dir)

    agg = pd.DataFrame({'pathway': pw_names, 'rd': 0.0})
    nc = 0
    for ki in range(K):
        key = f'crossmodal_proto_{ki}'
        if key not in analyzers: continue
        res = analyzers[key].rank_analysis()
        if 'entity' not in res.columns: continue
        rdm = dict(zip(res['entity'], res['rank_difference']))
        for i, pw in enumerate(pw_names):
            agg.loc[i, 'rd'] += rdm.get(pw, 0.0)
        nc += 1
    if nc == 0: return
    agg['rd'] /= nc

    hi = agg[agg['rd'] > 0].nlargest(n_dir, 'rd')['pathway'].tolist()
    lo = agg[agg['rd'] < 0].nsmallest(n_dir, 'rd')['pathway'].tolist()
    sel = hi + lo
    if not sel: return

    csv_dir = analysis_dir / 'crossmodal_gene_drilldown'
    csv_dir.mkdir(parents=True, exist_ok=True)
    agg[agg['pathway'].isin(sel)].sort_values('rd', key=abs, ascending=False).to_csv(
        str(csv_dir / 'selected_pathways.csv'), index=False)

    for pw in sel:
        if pw not in pw_names: continue
        pi = pw_names.index(pw)
        mask = np.zeros(len(g_names), dtype=bool)
        for p in valid:
            at = fold_attn[p]
            if 'gene_pathway_attention' not in at: continue
            mask |= (np.asarray(at['gene_pathway_attention'])[:, pi] > 0)
        part = [g_names[i] for i in range(len(g_names)) if mask[i]]
        gidx = [i for i in range(len(g_names)) if mask[i]]
        if len(part) < 2: continue

        safe = pw[:60].replace(' ', '_').replace('/', '_').replace(':', '_')
        nm = f'cm_genes_in_{safe}'
        an = FoldStratifiedAnalyzer(part, nm)
        for p in valid:
            at = fold_attn[p]
            if 'gene_pathway_attention' not in at: continue
            an.add_patient(p, np.asarray(at['gene_pathway_attention'])[:, pi][gidx],
                           risk_map[p], fold_idx)
        an.save_results(str(csv_dir))

        try:
            res = an.rank_analysis()
            if len(res) > 0:
                dirn = 'High Risk' if pw in hi else 'Low Risk'
                plot_rank_bars(res, f'Genes in {pw}\n(CM {dirn}, Fold {fold_idx})',
                               str(bar_dir / f'{nm}.pdf'), min(n_bar, len(res)))
        except Exception as e:
            logger.warning(f"    {pw}: {e}")


def plot_gating_comparison(fold_attn, risk_map, valid, K, fold_idx, output_path):
    """
    Rank-based prototype importance shift: how pathway context
    reshapes morphological priorities (Signal E → Signal H).

    For each patient, ranks prototypes by E and by H independently,
    then shows the mean rank change (H_rank - E_rank) per prototype.
    Positive = pathway context increased this prototype's importance.
    """
    from scipy.stats import rankdata

    COLOR_PROMOTED = '#4CAF50'   # green — pathway context boosted this
    COLOR_DEMOTED = '#9E9E9E'    # grey — pathway context suppressed this
    # F57C00 ORANGE

    e_ranks_all, h_ranks_all = [], []
    for p in valid:
        at = fold_attn[p]
        w_e = np.asarray(at['patch_assignments']['gate_weights'])[:K]
        w_h = np.asarray(at['fusion_gate_weights'])[:K]
        e_ranks_all.append(rankdata(w_e, method='average'))
        h_ranks_all.append(rankdata(w_h, method='average'))

    mean_e_rank = np.stack(e_ranks_all).mean(axis=0)
    mean_h_rank = np.stack(h_ranks_all).mean(axis=0)
    rank_delta = mean_h_rank - mean_e_rank  # positive = gained importance

    # Sort ascending (most demoted at top, most promoted at bottom)
    sort_idx = np.argsort(rank_delta)
    delta_sorted = rank_delta[sort_idx]
    labels = [f'Proto {i}' for i in sort_idx]
    colors = [COLOR_PROMOTED if d > 0 else COLOR_DEMOTED for d in delta_sorted]

    fig_height = max(6, K * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(K)
    ax.barh(y_pos, delta_sorted, color=colors, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color='grey', linewidth=0.8)
    ax.set_xlabel('Rank Shift (Fusion − WSI)', fontsize=11)
    ax.set_title(
        f'Pathway Context Effect on Prototype Importance — Fold {fold_idx}',
        fontsize=12, fontweight='bold',
    )
    ax.grid(axis='x', alpha=0.3)

    ax.legend(handles=[
        mpatches.Patch(facecolor=COLOR_PROMOTED, alpha=0.85, label='Rank increased'),
        mpatches.Patch(facecolor=COLOR_DEMOTED, alpha=0.85, label='Rank decreased'),
    ], loc='lower right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved gating rank comparison to {output_path}")


def plot_risk_distribution_by_event(fold_preds, fold_idx, output_path):
    """
    Histogram of risk scores coloured by event status (Image 2 style).

    Shows whether events cluster at the high-risk end of the score
    distribution — a basic sanity check for model calibration.
    """
    import matplotlib.pyplot as plt

    if 'risk_score_raw' in fold_preds.columns:
        scores = fold_preds['risk_score_raw'].values
        xlabel = 'Risk Score'
    elif 'risk_score' in fold_preds.columns:
        scores = fold_preds['risk_score'].values
        xlabel = 'Risk Score (rank-normalised)'
    else:
        logger.warning("  No risk scores for distribution plot")
        return

    events = fold_preds['event'].values.astype(int)
    n_event = int(events.sum())
    n_cens = int(len(events) - n_event)

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.histogram_bin_edges(scores, bins='auto')
    ax.hist(scores[events == 0], bins=bins, alpha=0.7,
            color='#5C6BC0', label=f'Censored (n={n_cens})', edgecolor='white')
    ax.hist(scores[events == 1], bins=bins, alpha=0.7,
            color='#EF5350', label=f'Event (n={n_event})', edgecolor='white')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Risk Score Distribution by Event Status',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved risk distribution to {output_path}")

# =====================================================================
# Plotting helpers
# =====================================================================

def plot_rank_bars(df, title, output_path, n=30, entity_col='entity',
                   metric_col='rank_difference', group_names=None,
                   figsize_per_row=0.35, min_fig_height=6):
    if group_names is None:
        group_names = {0: 'Low Risk', 1: 'High Risk'}
    df = df.copy()
    if metric_col not in df.columns: return

    # Top N from each direction
    high_risk = df[df[metric_col] > 0].nlargest(n // 2, metric_col)
    low_risk = df[df[metric_col] < 0].nsmallest(n // 2, metric_col)
    df = pd.concat([high_risk, low_risk]).sort_values(metric_col).reset_index(drop=True)

    h = max(min_fig_height, len(df) * figsize_per_row)

    fig, ax = plt.subplots(figsize=(10, h))
    colors = [COLOR_HIGH if v > 0 else COLOR_LOW for v in df[metric_col]]
    y = np.arange(len(df))
    ax.barh(y, df[metric_col], color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    labels = [str(l)[:55] + '...' if len(str(l)) > 55 else str(l) for l in df[entity_col]]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Rank Difference', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axvline(0, color='grey', lw=0.8); ax.grid(axis='x', alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(facecolor=COLOR_HIGH, alpha=0.85, label=f'Higher in {group_names[1]}'),
        mpatches.Patch(facecolor=COLOR_LOW, alpha=0.85, label=f'Higher in {group_names[0]}'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)


def create_all_bar_plots(analysis_dir, output_dir, n=30, group_names=None):
    os.makedirs(output_dir, exist_ok=True)
    title_map = {
        'pathway_gate': 'Pathway Gate Importance',
        'gene_average': 'Gene Importance (Average)',
        'gene_sum': 'Gene Importance (Sum)',
        'prototype_raw': 'Prototype Importance (WSI Gate)',
        'prototype_attended': 'Prototype Importance (Fusion Gate)',
        'prototype_assignment_freq': 'Prototype Assignment Frequency',
    }
    for f in sorted(os.listdir(analysis_dir)):
        if not f.endswith('_rank_analysis.csv'): continue
        nm = f.replace('_rank_analysis.csv', '')
        if nm.startswith('crossmodal_proto_') or nm.startswith('cm_genes_in_'):
            continue
        df = pd.read_csv(os.path.join(analysis_dir, f))
        if len(df) == 0: continue
        if nm in title_map: t = title_map[nm]
        elif nm.startswith('genes_in_'):
            t = f"Gene Attention in {nm.replace('genes_in_', '').replace('_', ' ')}"
        else: t = nm.replace('_', ' ').title()
        plot_rank_bars(df, t, os.path.join(output_dir, f'{nm}.pdf'), n, group_names=group_names)


def plot_crossmodal_summary_heatmap(
    analysis_dir,
    output_path,
    pathway_names,
    top_k_pathways=20,
    figsize=None,
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

    # Aggregate direction: mean rank_difference across prototypes
    pathway_mean_rd = {}
    for pw in all_pathways:
        pathway_mean_rd[pw] = np.mean([all_dfs[pi].get(pw, 0) for pi in proto_indices])

    # Top N per direction
    n_per_dir = top_k_pathways // 2
    all_pws = pd.Series(pathway_mean_rd)
    high_risk_pws = all_pws[all_pws > 0].nlargest(n_per_dir).index.tolist()
    low_risk_pws = all_pws[all_pws < 0].nsmallest(n_per_dir).index.tolist()
    top_pathways = high_risk_pws + low_risk_pws

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_figure(fig, output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved cross-modal heatmap to {output_path}")


def _select_top_pathways(analyzer, n_per_dir=5):
    res = analyzer.rank_analysis()
    hi = res[res['rank_difference'] > 0].nlargest(n_per_dir, 'rank_difference')['entity'].tolist()
    lo = res[res['rank_difference'] < 0].nsmallest(n_per_dir, 'rank_difference')['entity'].tolist()
    logger.info(f"  Selected {len(hi)} high + {len(lo)} low pathways for drill-down")
    return hi + lo


def _analysis_csvs_exist(analysis_dir, expected_names):
    """Check if rank_analysis CSVs exist for all expected signal names."""
    import os
    for name in expected_names:
        if not os.path.exists(
            os.path.join(str(analysis_dir), f'{name}_rank_analysis.csv')
        ):
            return False
    return True