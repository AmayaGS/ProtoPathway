"""
Visualization Entry Point for ProtoPathway.

Orchestrates the full post-evaluation interpretability pipeline:
    1. Pool CV fold data (predictions + attention)
    2. Kaplan-Meier survival curves
    3. Run importance analysis (all signal types)
    4. Generate bar plots, violin plots, shift plots
    5. Cross-modal attention heatmaps

Usage:
    python -m experiments.visualize --eval_dir results/TCGA-COADREAD/evaluation

    Or programmatically:
        from experiments.visualize import run_visualization
        run_visualization(eval_dir, entity_names, output_dir)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.analysis.fold_aggregation import pool_cv_results

from utils.analysis.importance_analyzer import (
    ImportanceAnalyzer,
    PrototypeShiftAnalyzer,
    run_importance_analysis,
)

from utils.analysis.fold_stratified_analysis import (
    run_fold_stratified_importance_analysis,
)

from utils.visualization.km_curves import plot_kaplan_meier_both
from utils.visualization.rank_bar_plots import (
    plot_rank_difference_bars,
    plot_top_differential_bars,
    create_all_bar_plots,
)
from utils.visualization.gate_violin_plots import (
    plot_gate_importance_violin,
    plot_assignment_frequency_violin,
)
from utils.visualization.prototype_shift import (
    plot_prototype_shift,
    plot_shift_slope,
)
from utils.visualization.cross_modal_heatmaps import (
    plot_cross_modal_comparison,
    plot_top_prototype_pathway_pairs,
)

logger = logging.getLogger(__name__)


def run_visualization(
        eval_dir: str,
        output_dir: Optional[str] = None,
        entity_names: Optional[Dict[str, List[str]]] = None,
        gene_idx_path: Optional[str] = None,
        pathway_idx_path: Optional[str] = None,
        pathways_of_interest: Optional[List[str]] = None,
        top_k_pathways: int = 10,
        risk_stratification: str = 'median',
        n_bar: int = 30,
        n_violin: int = 15,
):
    """
    Run the full visualization pipeline.

    Args:
        eval_dir: Path to evaluation directory with per-fold outputs.
        output_dir: Output directory for all figures and analysis CSVs.
            Defaults to eval_dir/../figures.
        entity_names: Dict with 'gene_names' and 'pathway_names' lists.
            If None, attempts to load from index files.
        gene_idx_path: Path to gene index file (if entity_names not given).
        pathway_idx_path: Path to pathway index file.
        pathways_of_interest: Specific pathways for within-pathway gene
            analysis. If None, auto-selects top-K from pathway gate analysis.
        top_k_pathways: Number of top pathways to auto-select.
        risk_stratification: 'median' or 'quartile'.
        n_bar: Number of entities in bar plots.
        n_violin: Number of entities in violin plots.
    """
    eval_dir = Path(eval_dir)

    if output_dir is None:
        output_dir = eval_dir.parent / 'figures'
    output_dir = Path(output_dir)

    # Create output subdirectories
    analysis_dir = output_dir / 'analysis'
    km_dir = output_dir / 'km_curves'
    bar_dir = output_dir / 'bar_plots'
    violin_dir = output_dir / 'violin_plots'
    shift_dir = output_dir / 'prototype_shift'
    heatmap_dir = output_dir / 'cross_modal'

    for d in [analysis_dir, km_dir, bar_dir, violin_dir, shift_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: Pool CV fold data
    # ================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Pooling CV fold data")
    logger.info("=" * 60)

    predictions, attention_by_patient, metadata = pool_cv_results(
        str(eval_dir), risk_stratification=risk_stratification
    )

    # Load entity names if not provided
    if entity_names is None:
        entity_names = {}

    if not entity_names.get('gene_names') and 'gene_names' in metadata:
        entity_names['gene_names'] = metadata['gene_names']
    if not entity_names.get('pathway_names') and 'pathway_names' in metadata:
        entity_names['pathway_names'] = metadata['pathway_names']

    gene_names = entity_names.get('gene_names', [])
    pathway_names = entity_names.get('pathway_names', [])
    n_protos = 16 # TODO get this from config at least...
    proto_names = [f'Prototype {i}' for i in range(n_protos)]

    logger.info(
        f"Data: {len(predictions)} patients, "
        f"{len(gene_names)} genes, {len(pathway_names)} pathways, "
        f"{n_protos} prototypes"
    )

    # ================================================================
    # Step 2: Kaplan-Meier curves
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Kaplan-Meier curves")
    logger.info("=" * 60)

    if 'risk_score' in predictions.columns:
        km_results = plot_kaplan_meier_both(
            times=predictions['survival_time'].values,
            events=predictions['event'].values,
            risk_scores=predictions['risk_score'].values,
            output_dir=str(km_dir),
        )
        for key, result in km_results.items():
            if 'p_value' in result:
                logger.info(f"  KM {key}: p = {result['p_value']:.4e}")
    else:
        logger.info("  Skipping KM curves (no risk scores)")

    # ================================================================
    # Step 3: Run importance analysis
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Importance analysis")
    logger.info("=" * 60)

    # analyzers = run_importance_analysis(
    #     predictions=predictions,
    #     attention_by_patient=attention_by_patient,
    #     entity_names=entity_names,
    #     output_dir=str(analysis_dir),
    #     pathways_of_interest=pathways_of_interest,
    #     top_k_pathways=top_k_pathways,
    # )

    analyzers = run_fold_stratified_importance_analysis(
        predictions=predictions,
        attention_by_patient=attention_by_patient,
        entity_names=entity_names,
        output_dir=str(analysis_dir),
        pathways_of_interest=pathways_of_interest,
        top_k_pathways=top_k_pathways,
    )

    # ================================================================
    # Step 4: Bar plots
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Bar plots")
    logger.info("=" * 60)

    create_all_bar_plots(
        analysis_dir=str(analysis_dir),
        output_dir=str(bar_dir),
        n=n_bar,
    )

    # ================================================================
    # Step 5: Violin plots
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Violin plots")
    logger.info("=" * 60)

    # Pathway gate importance violins
    if 'pathway_gate' in analyzers:
        plot_gate_importance_violin(
            analyzer=analyzers['pathway_gate'],
            title='Pathway Gate Importance by Risk Group',
            output_path=str(violin_dir / 'pathway_gate_violin.pdf'),
            n=n_violin,
            orientation='horizontal',
        )

    # Raw prototype importance violins
    if 'prototype_raw' in analyzers:
        plot_gate_importance_violin(
            analyzer=analyzers['prototype_raw'],
            title='Prototype Importance (WSI Gate) by Risk Group',
            output_path=str(violin_dir / 'prototype_raw_violin.pdf'),
            n=n_protos,
            orientation='vertical',
        )

    # Attended prototype importance violins
    if 'prototype_attended' in analyzers:
        plot_gate_importance_violin(
            analyzer=analyzers['prototype_attended'],
            title='Prototype Importance (After Pathway Attention) by Risk Group',
            output_path=str(violin_dir / 'prototype_attended_violin.pdf'),
            n=n_protos,
            orientation='vertical',
        )

    # Assignment frequency violins
    if 'prototype_assignment_freq' in analyzers:
        plot_assignment_frequency_violin(
            analyzer=analyzers['prototype_assignment_freq'],
            title='Prototype Assignment Frequency by Risk Group',
            output_path=str(violin_dir / 'assignment_frequency_violin.pdf'),
            n=n_protos,
        )

    # Gene average importance violins (top genes only)
    if 'gene_average' in analyzers:
        plot_gate_importance_violin(
            analyzer=analyzers['gene_average'],
            title='Gene Importance (Average) by Risk Group',
            output_path=str(violin_dir / 'gene_average_violin.pdf'),
            n=n_violin,
            orientation='horizontal',
        )

    # ================================================================
    # Step 6: Prototype shift plots
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Prototype shift plots")
    logger.info("=" * 60)

    if 'prototype_shift' in analyzers:
        shift_analyzer = analyzers['prototype_shift']

        plot_prototype_shift(
            shift_analyzer=shift_analyzer,
            output_path=str(shift_dir / 'prototype_shift.pdf'),
            show_by_risk=True,
        )

        plot_shift_slope(
            shift_analyzer=shift_analyzer,
            output_path=str(shift_dir / 'prototype_shift_slope.pdf'),
        )

    # ================================================================
    # Step 7: Cross-modal heatmaps
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Cross-modal attention heatmaps")
    logger.info("=" * 60)

    has_cross_modal = any(
        'cross_modal_attention' in data
        for data in attention_by_patient.values()
    )

    if has_cross_modal and pathway_names and proto_names:
        # Build significance mask from per-prototype rank analyses
        significance_mask = _build_significance_mask(analyzers, n_protos)

        plot_cross_modal_comparison(
            attention_by_patient=attention_by_patient,
            pathway_names=pathway_names,
            prototype_names=proto_names,
            output_dir=str(heatmap_dir),
            significance_mask=significance_mask,
        )

        plot_top_prototype_pathway_pairs(
            attention_by_patient=attention_by_patient,
            pathway_names=pathway_names,
            prototype_names=proto_names,
            output_path=str(heatmap_dir / 'top_pairs.pdf'),
            by_risk_group=True,
        )

    # ================================================================
    # Summary
    # ================================================================
    _print_summary(output_dir, analyzers)


def _build_significance_mask(
        analyzers: Dict,
        n_protos: int,
) -> Optional[Dict[int, np.ndarray]]:
    """
    Build significance mask for cross-modal heatmap overlay.

    Uses per-prototype pathway rank analyses to determine which
    prototype-pathway pairs show significantly different attention
    between risk groups.
    """
    mask = {}
    for proto_idx in range(n_protos):
        key = f'crossmodal_proto_{proto_idx}'
        if key not in analyzers:
            continue
        analyzer = analyzers[key]
        results = analyzer.rank_analysis()
        mask[proto_idx] = results['significant'].values

    return mask if mask else None


def _print_summary(output_dir: Path, analyzers: Dict):
    """Print final summary of generated outputs."""
    logger.info("\n" + "=" * 60)
    logger.info("Visualization complete!")
    logger.info("=" * 60)

    # Count output files by type
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        total_files += len(files)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total files generated: {total_files}")

    # Summarize significant findings
    sig_summary = []
    for name, analyzer in analyzers.items():
        if isinstance(analyzer, ImportanceAnalyzer):
            results = analyzer.rank_analysis()
            n_sig = results['significant'].sum()
            if n_sig > 0:
                top = results[results['significant']].head(3)
                entities = ', '.join(top['entity'].tolist())
                sig_summary.append(f"  {name}: {n_sig} significant — top: {entities}")
        elif isinstance(analyzer, PrototypeShiftAnalyzer):
            results = analyzer.analyze_shift()
            n_sig = results['significant'].sum()
            if n_sig > 0:
                top = results[results['significant']].head(3)
                protos = ', '.join(top['prototype'].tolist())
                sig_summary.append(f"  {name}: {n_sig} with significant shift — {protos}")

    if sig_summary:
        logger.info("\nSignificant findings:")
        for line in sig_summary:
            logger.info(line)
    else:
        logger.info("\nNo FDR-significant findings (may need larger cohort)")

    logger.info(f"\nDirectory structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n_files = sum(1 for _ in subdir.iterdir())
            logger.info(f"  {subdir.name}/  ({n_files} files)")