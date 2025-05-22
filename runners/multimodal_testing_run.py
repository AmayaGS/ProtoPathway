# runners/multimodal_testing_run.py

import os
import pickle
import pandas as pd

from utils.helpers import ensure_directory
from utils.model_utils import load_gene_expression_folds, load_wsi_folds
from train_test_loops.testers.multimodal_tester import MultimodalTester


def test_multimodal_model(config, is_continuation=False, is_full_train=False, experiment_logger=None):

    logger = experiment_logger.logger

    is_survival = config['execution'].get('task', 'classification') == 'survival'
    logger.info(f"Task type: {'survival' if is_survival else 'classification'}")

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

    logger.info(f"Starting multimodal model testing")
    logger.info(f"Results will be saved to {test_results_dir}")

    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)

    # Load the wsi data
    wsi_features_path = config['output']['data']['wsi_features']
    with open(wsi_features_path, "rb") as file:
        wsi_features = pickle.load(file)

    # Load data splits
    splits_dict_path = os.path.join(
        config['output']['data']['dir'],
        f"data_splits_{config['dataset_name']}.pkl"
    )

    with open(splits_dict_path, "rb") as f:
        split_dict = pickle.load(f)

    # Get test data
    if is_full_train:
        logger.info("Using train/test split for full ge_training")
        _, ge_test_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=False, ignore_missing=True)
        _, wsi_test_folds = load_wsi_folds(wsi_features, split_dict, is_cv=False, ignore_missing=True)
        n_folds = 1
    else:
        logger.info(f"Using {len(split_dict['CV'])} cross-validation folds")
        _, ge_test_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=True, ignore_missing=True)
        _, wsi_test_folds = load_wsi_folds(wsi_features, split_dict, is_cv=True, ignore_missing=True)
        n_folds = len(split_dict['CV'])

    fold_histories = []
    fold_summaries = []

    for fold_idx, (ge_test_data, wsi_test_data) in enumerate(zip(ge_test_folds, wsi_test_folds)):
        logger.info(f"Testing fold {fold_idx + 1}/{n_folds}")

        # Initialize tester
        tester = MultimodalTester(
            config=config,
            experiment_logger=experiment_logger,
            fold_idx=fold_idx
        )

        # Prepare test data
        ge_val_loader, wsi_val_loader = tester.prepare_test_data(ge_test_data, wsi_test_data)

        # Load the model
        checkpoint_name = f"best_fold_{fold_idx}.pt"
        model_path = os.path.join(model_dir, checkpoint_name)
        model = tester.load_model(model_path)

        # Test the model
        test_results = tester.test_model(model, ge_val_loader, wsi_val_loader)

        fold_histories.append(test_results['history'])
        fold_summaries.append(test_results['summary'])
