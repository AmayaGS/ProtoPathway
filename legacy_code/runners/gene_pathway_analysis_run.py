# runners/gene_pathway_analysis_run.py

import os
import pickle
import pandas as pd
from legacy_code.utils.helpers import ensure_directory
from legacy_code.utils.vis_results import GenePathwayAnalyzer


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


def run_gene_pathway_analysis(config, attention_dict, wsi_features, output_dir, experiment_logger=None):
    """Run comprehensive gene, pathway, and cross-modal analysis"""

    logger = experiment_logger.logger if experiment_logger else None
    ensure_directory(output_dir)

    patient_ids = [k for k in attention_dict.keys() if k not in ['gene_idx', 'pathway_idx']]

    if logger:
        logger.info(f"Analyzing {len(patient_ids)} patients")

    # Define analyzers
    analyzers = [
        ('gene', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'gene')),
        ('pathway', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'pathway')),
        ('crossmodal', GenePathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'], 'crossmodal'))
    ]

    summaries = {}

    # Run analysis for each type
    for analysis_name, analyzer in analyzers:
        # Add patient data
        for pid in patient_ids:
            label = wsi_features[pid][1]
            attn_data = (attention_dict[pid]['gene_pathway_attn'] if analysis_name != 'crossmodal'
                         else attention_dict[pid]['cross_modal_attn'])
            analyzer.add_patient(pid, attn_data, label)

        # Run complete analysis
        summaries[analysis_name] = _run_analysis(analyzer, output_dir, analysis_name, logger)

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

    return run_gene_pathway_analysis(None, attention_dict, wsi_features, output_dir, experiment_logger)