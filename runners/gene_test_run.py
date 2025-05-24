# runners/gene_test_run.py

import os
import pickle
import pandas as pd

from utils.helpers import ensure_directory
from train_test_loops.testers.gene_tester import GeneExpressionTester


def test_gene_expression_model(config, is_continuation=False, experiment_logger=None):
    """
    Test a trained gene expression model.

    Args:
        config: Configuration dictionary
        is_continuation: Whether this is continuing from a ge_training run
        experiment_logger: Logger instance for the experiment

    Returns:
        Dictionary with test results
    """
    logger = experiment_logger.logger

    # Set up paths
    if is_continuation:
        results_dir = experiment_logger.log_dir
        model_dir = experiment_logger.checkpoint_dir
    else:
        results_dir = config['testing']['experiment_path']
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Experiment directory not found: {results_dir}")

        model_dir = os.path.join(results_dir, "checkpoints")

    # Create a test results directory
    test_results_dir = os.path.join(results_dir, 'test_results')
    ensure_directory(test_results_dir)

    logger.info(f"Starting gene expression model testing")
    logger.info(f"Results will be saved to {test_results_dir}")

    # Load the gene expression data
    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)

    # Load data splits
    splits_dict_path = os.path.join(
        config['output']['data']['dir'],
        f"data_splits_{config['dataset_name']}.pkl"
    )

    with open(splits_dict_path, "rb") as f:
        split_dict = pickle.load(f)

    # Get test data
    test_data = gene_expression_df.loc[split_dict["Test"]]

    # Initialize tester
    tester = GeneExpressionTester(
        config=config,
        experiment_logger=experiment_logger,
        is_continuation=is_continuation
    )

    checkpoint_name = "best_fold_0.pt"

    model_path = os.path.join(model_dir, checkpoint_name)

    # Run testing
    test_results = tester.run_testing(test_data, checkpoint_path=model_path)

    # Log results
    accuracy = test_results['metrics']['accuracy'] / 100.0  # Convert to decimal
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_results['metrics']['f1']:.4f}")
    logger.info(f"Test AUC: {test_results['metrics']['auc']:.4f}")

    # If biomarker analysis was performed, log results
    if 'biomarker_analysis' in test_results:
        significant_pathways = len(test_results['biomarker_analysis']['biomarker_results']['pathway_biomarkers'][
                                       test_results['biomarker_analysis']['biomarker_results']['pathway_biomarkers'][
                                           'significant']
                                   ])
        significant_genes = len(test_results['biomarker_analysis']['biomarker_results']['gene_biomarkers'][
                                    test_results['biomarker_analysis']['biomarker_results']['gene_biomarkers'][
                                        'significant']
                                ])

        logger.info(f"Identified {significant_pathways} significant pathway biomarkers")
        logger.info(f"Identified {significant_genes} significant gene biomarkers")
        logger.info(
            f"Biomarker analysis report generated at: {test_results['biomarker_analysis']['report_paths']['report']}")

    return test_results