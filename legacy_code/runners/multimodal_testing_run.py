# runners/multimodal_testing_run.py

import os
import pickle
import pandas as pd

from legacy_code.utils.helpers import ensure_directory
from legacy_code.utils.model_utils import load_gene_expression_folds, load_wsi_folds


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

        # # Initialize tester
        # tester = MultimodalTester(
        #     config=config,
        #     experiment_logger=experiment_logger,
        #     fold_idx=fold_idx
        # )
        #
        # # Load the model
        # checkpoint_name = f"best_fold_{fold_idx}.pt"
        # model_path = os.path.join(model_dir, checkpoint_name)
        #
        # # Test the model
        # test_results = tester.run_testing(ge_test_data, wsi_test_data, model_path)
        # #
        # vis_results_path = r"C:\Users\Amaya\Documents\PhD\ProtoPathway\output\experiments\MM-MM-ProtoPathway_FT_fl-2-na_ds-R4RA_lr-1.0e-3_bs-1_dr-0.5_l1-0_l2-0_nl-0_hd-128_20250902_004932\visualise\vis_dict.pkl"
        # with open(vis_results_path, 'wb') as f:
        #     pickle.dump(test_results, f)

        if config['execution']['visualise']:
            from legacy_code.runners.gene_pathway_analysis_run import run_gene_pathway_analysis
            from legacy_code.utils.prototype_utils import generate_max_pathway_attention_heatmap
            # add prototype analysis runner functions here
            #
            vis_results_path = r"/output/experiments/MM-MM-ProtoPathway_FT_fl-2-na_ds-R4RA_lr-1.0e-3_bs-1_dr-0.5_l1-0_l2-0_nl-0_hd-128_20250902_004932/visualise/vis_dict.pkl"
            with open(vis_results_path, 'rb') as f:
                test_results = pickle.load(f)
            # #

            attention_dict = test_results['metrics']['attention_dict']
            predictions = test_results['metrics']['all_preds']
            run_gene_pathway_analysis(config, attention_dict, predictions, wsi_features, test_results_dir,
                                      experiment_logger)
            #
            # plots_dir = os.path.join(test_results_dir, 'plots')
            # create_plots(test_results_dir, plots_dir, config)

            # # Plot pathway gates from CSV
            # pathway_gates_csv = os.path.join(test_results_dir, 'pathway_gates_rank_differences.csv')
            # plot_pathway_gates_from_csv(
            #     csv_path=pathway_gates_csv,
            #     top_k=30,
            #     plot_type='rank_difference',
            #     output_path= os.path.join(plots_dir, 'pathway_gate_importance_plot.pdf')
            # )

            # HEATMAPS


            # ####################################################################################
            patient_ids = list(test_results['metrics']['attention_dict'].keys())[:-2]
            attention_dict = test_results['metrics']['attention_dict']
            gene_idx = attention_dict['gene_idx']
            pathway_idx = attention_dict['pathway_idx']

            gene_idx_inv = {v.item(): k for k, v in gene_idx.items()}
            pathway_idx_inv = {v.item(): k for k, v in pathway_idx.items()}

            for patient_id in patient_ids:
                patient_info = attention_dict[patient_id]
                cross_modal_attn = patient_info['cross_modal_attn']
                wsi_patch_names = wsi_features[patient_id][2]['filenames']
                patch_assignment = patient_info['hard_assignments'].squeeze(0)
                patch_assignment = [p.item() for p in patch_assignment]
                patch_coordinates = patient_info['patch_coords']
                path_to_patches = r"C:\Users\Amaya\Documents\PhD\Data\R4RA_patches\extracted_patches_2\extracted_patches.csv"
                output_dir = os.path.join(test_results_dir, 'prototype_plots')
                ensure_directory(output_dir)
            #
                generate_max_pathway_attention_heatmap(
                    patient_id=patient_id,
                    patch_assignments=patch_assignment,
                    patch_names=wsi_patch_names,
                    patch_coordinates=patch_coordinates,
                    cross_modal_attn=cross_modal_attn,
                    pathway_names=pathway_idx_inv,
                    extracted_patches_path=path_to_patches,
                    output_dir=output_dir,
                    fold=fold_idx,
                    patch_size=224,
                    show_values=False,
                    use_bin=False

                )

                # analyze_prototype_distribution(
                #     patient_id=patient_id,
                #     patch_assignments=patch_assignment,
                #     output_dir=output_dir,
                #     use_binning = True
                # )

                # generate_prototype_heatmap(
                #     patient_id=patient_id,
                #     patch_assignments=patch_assignment,
                #     patch_names=wsi_patch_names,
                #     patch_coordinates=patch_coordinates,
                #     extracted_patches_path=path_to_patches,
                #     output_dir=output_dir,
                #     fold=0,
                #     patch_size=224,
                #     use_binning=False
                # )
                #
                # generate_prototype_heatmap(
                #     patient_id=patient_id,
                #     patch_assignments=patch_assignment,
                #     patch_names=wsi_patch_names,
                #     patch_coordinates=patch_coordinates,
                #     extracted_patches_path=path_to_patches,
                #     output_dir=output_dir,
                #     fold=0,
                #     patch_size=224
                # )






