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

logger = logging.getLogger(__name__)

COLOR_LOW = '#2196F3'
COLOR_HIGH = '#E53935'


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
    # Spatial (Step 6)
    wsi_features_dir: Optional[str] = None,
    wsi_dir: Optional[str] = None,
    spatial_fold: Optional[int] = None,
    spatial_patient: Optional[str] = None,
    spatial_n_per_group: int = 2,
    spatial_downsample: int = 4,
    spatial_patch_size: int = 256,
    spatial_single_pathway: Optional[str] = None,
):
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

    # ── Step 2 ───────────────────────────────────────────────────────
    logger.info("Step 2: Kaplan-Meier curves")
    if 'risk_score' in predictions.columns:
        plot_kaplan_meier_both(
            times=predictions['survival_time'].values,
            events=predictions['event'].values,
            risk_scores=predictions['risk_score'].values,
            output_dir=str(km_dir),
        )

    # ── Step 3 ───────────────────────────────────────────────────────
    logger.info("Step 3: Cross-fold importance analysis")
    if use_fold_stratified:
        analyzers = run_fold_stratified_importance_analysis(
            predictions=predictions, attention_by_patient=attention_by_patient,
            entity_names=entity_names, output_dir=str(analysis_dir),
            pathways_of_interest=[], top_k_pathways=0,
            skip_prototype_signals=True,
        )
    else:
        analyzers = run_importance_analysis(
            predictions=predictions, attention_by_patient=attention_by_patient,
            entity_names=entity_names, output_dir=str(analysis_dir),
            pathways_of_interest=[], top_k_pathways=0,
        )

    pathways_of_interest = []
    if 'pathway_gate' in analyzers:
        pathways_of_interest = _select_top_pathways(
            analyzers['pathway_gate'], n_pathways_per_direction)

    if pathways_of_interest:
        logger.info(f"Step 3b: Gene drill-down ({len(pathways_of_interest)} pathways)")
        if use_fold_stratified:
            gene_a = run_fold_stratified_importance_analysis(
                predictions=predictions, attention_by_patient=attention_by_patient,
                entity_names=entity_names, output_dir=str(analysis_dir),
                pathways_of_interest=pathways_of_interest, top_k_pathways=0,
                skip_prototype_signals=True,
            )
        else:
            gene_a = run_importance_analysis(
                predictions=predictions, attention_by_patient=attention_by_patient,
                entity_names=entity_names, output_dir=str(analysis_dir),
                pathways_of_interest=pathways_of_interest, top_k_pathways=0,
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
):
    from pathlib import Path
    from utils.visualization.spatial_heatmaps import generate_patient_spatial_viz
    from utils.visualization.prototype_panels import (
        plot_prototype_importance, plot_cohort_prototype_exemplars,
    )

    output_dir = Path(output_dir)
    spatial_dir = output_dir / 'spatial'
    spatial_dir.mkdir(parents=True, exist_ok=True)

    fold_analysis_dir = perfold_dir / f'fold_{fold_idx}' / 'analysis'
    pathway_names = entity_names.get('pathway_names', [])

    if not fold_analysis_dir.exists():
        logger.error(f"  Per-fold analysis not found: {fold_analysis_dir}")
        return

    # Select patients
    if patient_id:
        pids = [patient_id]
    else:
        pids = _auto_select_patients(
            predictions, attention_by_patient, wsi_features_dir, n_per_group)

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
                fold_analysis_dir=str(fold_analysis_dir),
                output_dir=str(spatial_dir / pid),
                risk_group=rg,
                wsi_features_dir=wsi_features_dir,
                wsi_dir=wsi_dir,
                patch_size=patch_size, downsample=downsample,
                single_pathway_name=single_pathway_name,
                rank_transform=True,
            )
        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Cohort prototype panels
    logger.info("\n  Cohort prototype panels...")
    proto_dir = spatial_dir / 'prototype_panels'
    try:
        plot_prototype_importance(
            attention_by_patient=attention_by_patient,
            output_dir=str(proto_dir), top_k=5, dpi=300,
        )
    except Exception as e:
        logger.error(f"  Prototype importance failed: {e}")

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

    logger.info(f"  Spatial outputs in {spatial_dir}")


def _auto_select_patients(predictions, attention_by_patient, wsi_features_dir, n):
    import torch
    available = predictions[
        predictions['patient_id'].isin(attention_by_patient.keys())
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
        plot_gate_importance_violin, plot_assignment_frequency_violin,
    )
    from utils.visualization.prototype_shift import (
        plot_prototype_shift, plot_shift_slope,
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

    if has_e:
        logger.info(f"  Signal E (WSI gate)...")
        _build('prototype_raw', pnames,
               lambda a: a['patch_assignments']['gate_weights'])

    if has_h:
        logger.info(f"  Signal H (fusion gate)...")
        _build('prototype_attended', pnames,
               lambda a: a['fusion_gate_weights'])

    if has_e and has_h:
        logger.info(f"  Shift E→H...")
        sa = PrototypeShiftAnalyzer(pnames)
        for p in valid:
            at = fold_attn[p]
            sa.add_patient(p, wsi_gate=at['patch_assignments']['gate_weights'],
                           fusion_gate=at['fusion_gate_weights'],
                           risk_group=risk_map[p])
        sa.save_results(str(dirs['analysis']))
        analyzers['prototype_shift'] = sa

    if has_e and 'assignments' in sample.get('patch_assignments', {}):
        logger.info(f"  Signal F (assignment freq)...")
        fa = FoldStratifiedAnalyzer(pnames, 'prototype_assignment_freq')
        for p in valid:
            assigns = np.asarray(fold_attn[p]['patch_assignments']['assignments']).astype(int)
            if len(assigns) == 0: continue
            freq = np.array([(assigns == i).sum() / len(assigns) for i in range(K)])
            fa.add_patient(p, freq, risk_map[p], fold_idx)
        fa.save_results(str(dirs['analysis']))
        analyzers['prototype_assignment_freq'] = fa

    if has_g and pw_names:
        logger.info(f"  Signal G (cross-modal, {K}×{len(pw_names)})...")
        for ki in range(K):
            nm = f'crossmodal_proto_{ki}'
            a = FoldStratifiedAnalyzer(pw_names, nm)
            for p in valid:
                a.add_patient(p, fold_attn[p]['cross_modal_attention'][ki],
                              risk_map[p], fold_idx)
            a.save_results(str(dirs['analysis']))
            analyzers[nm] = a

    # ── Bar plots ────────────────────────────────────────────────────
    logger.info(f"  Bar plots...")
    create_all_bar_plots(str(dirs['analysis']), str(dirs['bar_plots']), n_bar)

    # ── Violin plots ─────────────────────────────────────────────────
    logger.info(f"  Violin plots...")
    for key, title, n_e, ori in [
        ('prototype_raw', f'WSI Gate — Fold {fold_idx}', K, 'vertical'),
        ('prototype_attended', f'Fusion Gate — Fold {fold_idx}', K, 'vertical'),
    ]:
        if key in analyzers:
            try:
                plot_gate_importance_violin(
                    analyzer=analyzers[key], title=title,
                    output_path=str(dirs['violin_plots'] / f'{key}_violin.pdf'),
                    n=n_e, orientation=ori)
            except Exception as e:
                logger.warning(f"  {key} violin: {e}")

    if 'prototype_assignment_freq' in analyzers:
        try:
            plot_assignment_frequency_violin(
                analyzer=analyzers['prototype_assignment_freq'],
                title=f'Assignment Frequency — Fold {fold_idx}',
                output_path=str(dirs['violin_plots'] / 'assignment_freq_violin.pdf'),
                n=K)
        except Exception as e:
            logger.warning(f"  Freq violin: {e}")

    # ── Shift plots ──────────────────────────────────────────────────
    if 'prototype_shift' in analyzers:
        logger.info(f"  Shift plots...")
        sa = analyzers['prototype_shift']
        try:
            plot_prototype_shift(sa, str(dirs['shift_plots'] / 'shift.pdf'),
                                 title=f'Prototype Shift — Fold {fold_idx}',
                                 show_by_risk=True)
        except Exception as e:
            logger.warning(f"  Shift bars: {e}")
        try:
            plot_shift_slope(sa, str(dirs['shift_plots'] / 'shift_slope.pdf'),
                             title=f'Before vs After — Fold {fold_idx}')
        except Exception as e:
            logger.warning(f"  Slope: {e}")

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
    df['_abs'] = df[metric_col].abs()
    df = df.nlargest(n, '_abs').sort_values(metric_col).reset_index(drop=True)

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
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)


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


def plot_crossmodal_summary_heatmap(analysis_dir, output_path, pw_names=None,
                                     top_k=20, figsize=None):
    csvs = sorted(f for f in os.listdir(analysis_dir)
                  if f.startswith('crossmodal_proto_') and f.endswith('_rank_analysis.csv'))
    if not csvs: return
    dfs = {}
    for f in csvs:
        pi = int(f.split('_')[-2])
        df = pd.read_csv(os.path.join(analysis_dir, f))
        dfs[pi] = dict(zip(df['entity'], df['rank_difference']))
    pis = sorted(dfs.keys())
    pws = set()
    for d in dfs.values(): pws.update(d.keys())
    top = sorted(pws, key=lambda pw: max(abs(dfs[p].get(pw, 0)) for p in pis), reverse=True)[:top_k]
    M = np.zeros((len(pis), len(top)))
    for i, p in enumerate(pis):
        for j, pw in enumerate(top):
            M[i, j] = dfs[p].get(pw, 0)
    disp = [pw[:50] + '...' if len(pw) > 50 else pw for pw in top]
    if figsize is None:
        figsize = (max(12, len(top) * 0.5), max(6, len(pis) * 0.6))
    fig, ax = plt.subplots(figsize=figsize)
    vm = max(abs(M.min()), abs(M.max())) or 1.0
    im = ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(-vm, 0, vm), aspect='auto')
    ax.set_xticks(np.arange(len(top))); ax.set_xticklabels(disp, fontsize=7, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(pis))); ax.set_yticklabels([f'Proto {i}' for i in pis], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Rank Diff', fontsize=9)
    ax.set_title('Cross-Modal Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = _ensure_pdf(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)


def _select_top_pathways(analyzer, n_per_dir=5):
    res = analyzer.rank_analysis()
    hi = res[res['rank_difference'] > 0].nlargest(n_per_dir, 'rank_difference')['entity'].tolist()
    lo = res[res['rank_difference'] < 0].nsmallest(n_per_dir, 'rank_difference')['entity'].tolist()
    logger.info(f"  Selected {len(hi)} high + {len(lo)} low pathways for drill-down")
    return hi + lo


def _ensure_pdf(path):
    b, e = os.path.splitext(str(path))
    return b + '.pdf' if e.lower() != '.pdf' else str(path)