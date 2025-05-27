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



    for fold_idx, (ge_test_data, wsi_test_data) in enumerate(zip(ge_test_folds, wsi_test_folds)):
        logger.info(f"Testing fold {fold_idx + 1}/{n_folds}")

        # Initialize tester
        tester = MultimodalTester(
            config=config,
            experiment_logger=experiment_logger,
            fold_idx=fold_idx
        )

        # Load the model
        checkpoint_name = f"best_fold_{fold_idx}.pt"
        model_path = os.path.join(model_dir, checkpoint_name)

        # Test the model
        # test_results = tester.run_testing(ge_test_data, wsi_test_data, model_path)

        if config['execution']['visualise']:
            from runners.gene_pathway_analysis_run import run_gene_pathway_analysis
            from utils.simple_visualizations import create_plots

            # vis_results = r"C:\Users\Amaya\Documents\PhD\ProtoPathway\output\experiments\MM-MM-ProtoPathway_FT_fl-2-128_ds-R4RA_lr-1.0e-5_bs-1_dr-0.2_l1-0_l2-0_nl-0_hd-128_20250525_180450\visualise\vis_dict.pkl"
            # with open(vis_results, 'rb') as f:
            #     test_results = pickle.load(f)
            #
            # attention_dict = test_results['metrics']['attention_dict']
            # run_gene_pathway_analysis(config, attention_dict, wsi_features, test_results_dir, experiment_logger)

            plots_dir = os.path.join(test_results_dir, 'plots')
            create_plots(test_results_dir, plots_dir, config)

            # ####################################################################################
            # patient_ids = list(test_results['metrics']['attention_dict'].keys())[-2]
            # gene_idx = attention_dict['gene_idx']
            # pathway_idx = attention_dict['pathway_idx']
            #
            # gene_idx_inv = {v.item(): k for k, v in gene_idx.items()}
            # pathway_idx_inv = {v.item(): k for k, v in pathway_idx.items()}
            #
            # for patient_id in patient_ids:
            #     patient_info = attention_dict[patient_id]
            #     wsi_patch_names = wsi_features[patient_id][2]['filenames']
            #     patch_asssignment = patient_info['hard_assignments'].squeeze(0)
            #     patch_asssignment = [p.item() for p in patch_asssignment]
            #     sorted_gene_importance = patient_info['gene_pathway_attn'].sum(dim=1).sort()





