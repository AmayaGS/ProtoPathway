# runners/gene_pathway_analysis_run.py

import os
import pickle
import pandas as pd
from utils.helpers import ensure_directory
from utils.vis_results import GenePathwayAnalyzer
from utils.vis_results import PathwayGateAnalyzer
from utils.vis_results import PrototypeGateAnalyzer
from utils.vis_results import plot_prototype_importance, plot_prototype_importance_by_class
from utils.vis_results import PrototypePatchAnalyzer


def _run_analysis(analyzer, output_dir, analysis_name, logger=None):
    """Run complete analysis pipeline for any analyzer type"""

    # Individual patient results
    patient_dir = os.path.join(output_dir, f'patient_{analysis_name}_importance')
    analyzer.save_patient_results(patient_dir)

    # Class aggregation
    class_results = analyzer.class_aggregation(k=10000)
    for label, df in class_results.items():
        df.to_csv(os.path.join(output_dir, f'class_{label}_top_{analyzer.entity_name}s.csv'), index=False)

    # Statistical differences
    diff_results = analyzer.class_differences(k=10000)
    diff_results.to_csv(os.path.join(output_dir, f'{analysis_name}_differences.csv'), index=False)

    # Class-specific drivers
    drivers = analyzer.class_specific_drivers(k=10000)
    for driver_type, df in drivers.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(output_dir, f'{driver_type}_{analysis_name}_{analyzer.entity_name}s.csv'), index=False)

    # Rank-based analysis
    rank_results = analyzer.rank_based_analysis(k=10000)
    rank_results.to_csv(os.path.join(output_dir, f'{analysis_name}_ranks.csv'), index=False)

    # Consensus analysis
    analyzer.consensus_analysis(output_dir, k_per_method=10000)

    if logger:
        logger.info(f"{analysis_name.title()} analysis: {drivers['summary']}")

    return drivers['summary']


def run_gene_pathway_analysis(config, attention_dict, predictions, wsi_features, output_dir, experiment_logger=None):
    """Run comprehensive gene, pathway, and cross-modal analysis

    Args:
        predictions:
    """

    logger = experiment_logger.logger if experiment_logger else None
    ensure_directory(output_dir)

    patient_ids = [k for k in attention_dict.keys() if k not in ['gene_idx', 'pathway_idx']]

    if logger:
        logger.info(f"Analyzing {len(patient_ids)} patients")

    # # Define analyzers
    # analyzers = [
    #     ('gene_sum', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'gene_sum')),
    #     ('gene_average', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'gene_average'))
    #     # ('pathway', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'pathway')),
    #     # ('crossmodal', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'crossmodal'))
    # ]
    #
    # summaries = {}
    #
    # # Run analysis for each type
    # for analysis_name, analyzer in analyzers:
    #     # Add patient data
    #     for pid in patient_ids:
    #         label = wsi_features[pid][1]
    #         attn_data = (attention_dict[pid]['gene_pathway_attn'] if analysis_name != 'crossmodal'
    #                      else attention_dict[pid]['cross_modal_attn'])
    #         analyzer.add_patient(pid, attn_data, label)
    #
    #     # Run complete analysis
    #     summaries[analysis_name] = _run_analysis(analyzer, output_dir, analysis_name, logger)
    #
    # gene_sum_analyzer = analyzers[0][1]
    #
    # pathway_stats = gene_sum_analyzer.plot_pathway_gene_class_differences(
    #     pathway_name="AMPK inhibits chREBP transcriptional activation activity",
    #     top_k=8,
    #     plot_type='difference',
    #     output_path=os.path.join(output_dir, 'plots/pathway_gene_differences.pdf')
    # )

    # # 🔥 ADD PATHWAY GATE ANALYSIS
    # logger.info("Analyzing pathway gating weights...")
    #
    # pathway_gate_analyzer = PathwayGateAnalyzer(attention_dict['pathway_idx'])
    #
    # # Add patient data to pathway gate analyzer
    # for pid in patient_ids:
    #     label = wsi_features[pid][1]
    #     pathway_gates = attention_dict[pid]['pathway_attn_softmax']
    #     pathway_gate_analyzer.add_patient(pid, pathway_gates, label)
    #
    # pathway_gate_results = pathway_gate_analyzer.pathway_rank_differences(k=200)
    #
    # if pathway_gate_results is not None:
    #     # Save to CSV (simple!)
    #     pathway_gate_results.to_csv(os.path.join(output_dir, 'pathway_gates_rank_differences.csv'), index=False)
    #
    #     if logger:
    #         top_pathway = pathway_gate_results.iloc[0]
    #         logger.info(
    #             f"Top pathway rank difference: {top_pathway['pathway']} (diff: {top_pathway['rank_difference']:.1f})")
    #
    # # Generate pathway gate plots
    # gate_plot_path = os.path.join(output_dir, 'pathway_gating_weights_class_comparison.pdf')
    # gate_stats = pathway_gate_analyzer.plot_pathway_class_differences(
    #     top_k=20,
    #     output_path=gate_plot_path,
    #     plot_type='difference'
    # )
    #
    # gate_plot_path = os.path.join(output_dir, 'pathway_gating_weights_class_comparison.pdf')
    # gate_stats = pathway_gate_analyzer.plot_statistical_pathway_differences(
    #     top_k=30,
    #     plot_type='rank_difference',  # or 'fold_change' or 'rank_difference'
    #     output_path=gate_plot_path
    # )
    #
    # # Save pathway gate class results
    # gate_class_results = pathway_gate_analyzer.class_aggregation(k=15)
    # for label, df in gate_class_results.items():
    #     df.to_csv(os.path.join(output_dir, f'class_{label}_top_pathway_gates.csv'), index=False)
    #
    # logger.info(f"Top pathway gate difference: {gate_stats['max_difference_pathway']}")

    # # # 🔥 PROTOTYPE GATE ANALYSIS
    # prototype_gate_analyzer = PrototypeGateAnalyzer(num_prototypes=64)  # Adjust number
    # # #
    # # Add patient data
    # for pid in patient_ids:
    #     label = wsi_features[pid][1]
    #     prototype_gates = attention_dict[pid]['gate_importance']  # Make sure this exists!
    #     prototype_gate_analyzer.add_patient(pid, prototype_gates, label)
    #
    # # Generate global importance
    # global_prototype_results = prototype_gate_analyzer.global_prototype_importance(k=20)
    # if global_prototype_results is not None:
    #     global_prototype_results.to_csv(os.path.join(output_dir, 'prototype_gates_global_importance.csv'), index=False)
    #
    # # Plot global importance
    # plot_prototype_importance(
    #     global_prototype_results,
    #     plot_type='global',
    #     top_k=15,
    #     output_path=os.path.join(output_dir, 'prototype_gates_global.png')
    # )
    #
    # highest_attention_protos = global_prototype_results.head(20)['prototype_id'].tolist()
    #
    # violin_data_colorful = prototype_gate_analyzer.plot_prototype_violin(
    #     prototype_ids=highest_attention_protos,
    #     figsize=(18, 8),
    #     output_path=os.path.join(output_dir, 'prototype_importance_violins_discriminative.pdf')
    # )
    #
    # prototype_rank_results = prototype_gate_analyzer.prototype_rank_differences(sort_by="rank_difference", k=20)['prototype_id'].tolist()
    # violin_data_colorful = prototype_gate_analyzer.plot_prototype_violin(
    #     prototype_ids=prototype_rank_results,
    #     figsize=(18, 8),
    #     output_path=os.path.join(output_dir, 'prototype_importance_violins_rank_relevance.pdf')
    # )

    # # Generate rank differences
    # prototype_rank_results = prototype_gate_analyzer.prototype_rank_differences(sort_by="rank_difference", k=20)
    # if prototype_rank_results is not None:
    #     prototype_rank_results.to_csv(os.path.join(output_dir, 'prototype_gates_rank_differences.csv'), index=False)
    # #
    #     # Plot rank differences
    #     plot_prototype_importance(
    #         prototype_rank_results,
    #         plot_type='rank_difference',
    #         top_k=15,
    #         output_path=os.path.join(output_dir, 'prototype_gates_differences.png')
    #     )
    # #
    #     # Generate rank differences
    #     prototype_rank_results = prototype_gate_analyzer.prototype_rank_differences(k=20)
    #     if prototype_rank_results is not None:
    #         prototype_rank_results.to_csv(os.path.join(output_dir, 'prototype_gates_percent_differences.csv'), index=False)
    #
    #     # Plot rank differences
    #     plot_prototype_importance(
    #         prototype_rank_results,
    #         plot_type='percentage_difference',
    #         top_k=15,
    #         output_path=os.path.join(output_dir, 'prototype_gates_percentage_difference.png')
    #     )

    # Generate prototype differences sorted by percentage
    # prototype_pct_results = prototype_gate_analyzer.prototype_rank_differences(
    #     k=60,
    #     sort_by='percentage_difference'
    # )
    # prototype_pct_results.to_csv(os.path.join(output_dir, 'prototype_gates_rank_differences.csv'),
    #                              index=False)
    #
    # # Plot (now ordered by percentage difference magnitude)
    # plot_prototype_importance(
    #     prototype_pct_results,
    #     plot_type='percentage_difference',
    #     top_k=20,
    #     output_path=os.path.join(output_dir, 'prototype_percentage_differences.pdf')
    # )

    # prototype_results = prototype_gate_analyzer.prototype_rank_differences(k=50)  # Get more data

    # # Create side-by-side class comparison
    # class_comparison = plot_prototype_importance_by_class(
    #     prototype_pct_results,
    #     top_k=15,
    #     output_path=os.path.join(output_dir, 'prototype_importance_by_class.pdf')
    # )
    #
    prototype_gate_analyzer = PrototypeGateAnalyzer(num_prototypes=64)  # Adjust number
    #
    # Add patient data
    for pid in patient_ids:
        label = wsi_features[pid][1]
        prototype_gates = attention_dict[pid]['gate_importance']  # Make sure this exists!
        prototype_gate_analyzer.add_patient(pid, prototype_gates, label)

    # Generate prototype differences sorted by percentage
    prototype_pct_results = prototype_gate_analyzer.prototype_rank_differences(
        k=60,
        sort_by='rank_difference'
    )
    #
    # global_prototype_results = prototype_gate_analyzer.global_prototype_importance(k=20)
    # highest_attention_protos = global_prototype_results.head(6)['prototype_id'].tolist()
    #
    # Initialize patch analyzer
    patch_analyzer = PrototypePatchAnalyzer(
        extracted_patches_path=r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches\extracted_patches_2\extracted_patches.csv"
    )

    # Get most discriminative prototypes from your gate analysis
    discriminative_protos = patch_analyzer.get_discriminative_prototypes(
        prototype_pct_results,  # From your PrototypeGateAnalyzer
        top_k=20
    )

    for i, pid in enumerate(patient_ids):
        patient_info = attention_dict[pid]
        patch_assignments = patient_info['hard_assignments']
        soft_assignments = patient_info['soft_assignments']
        similarities = patient_info['similarities']  # 🔥 Use similarity matrix!
        patch_names = wsi_features[pid][2]['filenames']
        class_label = wsi_features[pid][1]
        prediction = predictions[i]
        patch_analyzer.add_patient_data(pid, patch_names, patch_assignments, similarities, class_label, prediction)

    # Generate the threshold analyses:
    patch_analyzer.analyze_prototype_thresholds(prototype_ids=[25, 0, 21, 50], output_path=os.path.join(output_dir,"prototype_highest_attended_thresholds.pdf"))
    patch_analyzer.plot_assignment_scatter(prototype_id=25, output_path="p25_scatter.png")
    patch_analyzer.plot_diagnostic_thresholds_summary(key_prototypes=[25, 0, 21, 58], output_path=os.path.join(output_dir,"threshold_summary.pdf"))


    # pct_diff = [28, 3, 48, 10, 27] # most percentage difference prototypes
    # pct_diff = [57, 43, 12, 15, 36]
    pct_diff = [25, 0, 21, 4, 50]  # most rank difference prototypes
    # Plot with intelligent selection
    patch_analyzer.plot_prototype_patches(
        prototype_ids=pct_diff,
        selection_method='top_similarity',  # 🔥 Show most representative patches!
        patches_per_prototype=6,
        patches_per_class=3,
        output_path=os.path.join(output_dir, 'percentage_difference_prototype_patches.pdf')
    )

    patch_analyzer.plot_prototype_patches(
        prototype_ids=highest_attention_protos[:-1],
        selection_method='top_similarity',
        patches_per_prototype=6,
        patches_per_class=3,
        output_path=os.path.join(output_dir, 'highest_attention_prototype_patches.pdf')
    )


    # # Create summary report
    # summary = patch_analyzer.create_prototype_summary_report(
    #     discriminative_protos,
    #     output_dir
    # )

    # Save summary
    summary_df = pd.DataFrame([{'total_patients': len(patient_ids), **summaries}])
    summary_df.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)

    if logger:
        logger.info("Analysis complete!")

    return summaries


def load_and_run_analysis(vis_results_path, wsi_features_path, output_dir, experiment_logger=None):
    """Load data and run analysis"""

    with open(vis_results_path, 'rb') as f:
        test_results = pickle.load(f)

    with open(wsi_features_path, 'rb') as f:
        wsi_features = pickle.load(f)

    attention_dict = test_results['metrics']['attention_dict']

    return run_gene_pathway_analysis(None, attention_dict, predictions, wsi_features, output_dir, experiment_logger)