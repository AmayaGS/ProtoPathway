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
            import numpy as np
            from utils.vis_results import GeneImportanceAnalyzer, PathwayImportanceAnalyzer, CrossModalPathwayAnalyzer

            vis_results = r"C:\Users\Amaya\Documents\PhD\ProtoPathway\output\experiments\MM-MM-ProtoPathway_FT_fl-2-128_ds-R4RA_lr-1.0e-5_bs-1_dr-0.2_l1-0_l2-0_nl-0_hd-128_20250525_180450\visualise\vis_dict.pkl"
            with open(vis_results, 'rb') as f:
                test_results = pickle.load(f)

            attention_dict = test_results['metrics']['attention_dict']
            analyzer = GeneImportanceAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'])

            patient_ids = [k for k in attention_dict.keys() if k not in ['gene_idx', 'pathway_idx']]

            # === GENE-LEVEL ANALYSIS ===
            # for pid in patient_ids:
            #     label = wsi_features[pid][1]
            #     analyzer.add_patient(pid, attention_dict[pid]['gene_pathway_attn'], label)
            #
            # patient_output_dir = os.path.join(test_results_dir, 'patient_gene_importance')
            # analyzer.save_patient_results(patient_output_dir)
            #
            # class_results = analyzer.class_aggregation(k=500)
            # for label, df in class_results.items():
            #     df.to_csv(os.path.join(test_results_dir, f'class_{label}_top_genes.csv'), index=False)
            #     logger.info(f"Class {label} top 10:\n{df.head(10)}")
            #
            # # Differential analysis - shows which genes differ most between classes
            # diff_results = analyzer.class_differences(k=500)
            # diff_results.to_csv(os.path.join(test_results_dir, 'class_differences.csv'), index=False)
            # significant_genes = diff_results[diff_results['significant'] == True]
            # logger.info(f"Found {len(significant_genes)} statistically significant differential genes")
            # logger.info(f"Top 10 differential genes:\n{diff_results.head(10)}")
            #
            # # Class-specific drivers - separates genes by which class they drive
            # class_drivers = analyzer.class_specific_drivers(k=100)
            # for driver_type, df in class_drivers.items():
            #     driver_file = os.path.join(test_results_dir, f'{driver_type}.csv')
            #     df.to_csv(driver_file, index=False)
            #     logger.info(f"Top 5 {driver_type}:\n{df.head(5)}")
            #
            # # === PATHWAY-LEVEL ANALYSIS ===
            # pathway_analyzer = PathwayImportanceAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'])
            #
            # for pid in patient_ids:
            #     label = wsi_features[pid][1]
            #     pathway_analyzer.add_patient(pid, attention_dict[pid]['gene_pathway_attn'], label)
            #
            # # Save individual pathway results
            # pathway_output_dir = os.path.join(test_results_dir, 'patient_pathway_importance')
            # pathway_analyzer.save_patient_results(pathway_output_dir)
            #
            # # Pathway class aggregation
            # pathway_class_results = pathway_analyzer.class_aggregation(k=500)
            # for label, df in pathway_class_results.items():
            #     df.to_csv(os.path.join(test_results_dir, f'class_{label}_top_pathways.csv'), index=False)
            #     logger.info(f"Class {label} top 5 pathways:\n{df.head(5)}")
            #
            # # Pathway differential analysis
            # pathway_diff_results = pathway_analyzer.class_differences(k=100)
            # pathway_diff_results.to_csv(os.path.join(test_results_dir, 'pathway_differences.csv'), index=False)
            # pathway_significant = pathway_diff_results[pathway_diff_results['significant'] == True]
            # logger.info(f"Found {len(pathway_significant)} significant differential pathways")
            # logger.info(f"Top 5 differential pathways:\n{pathway_diff_results.head(5)}")
            #
            # # Pathway class-specific drivers
            # pathway_drivers = pathway_analyzer.class_specific_drivers(k=100)
            # for driver_type, df in pathway_drivers.items():
            #     driver_file = os.path.join(test_results_dir, f'{driver_type}_pathways.csv')
            #     df.to_csv(driver_file, index=False)
            #     logger.info(f"Top 3 {driver_type} pathways:\n{df.head(3)}")
            #
            # logger.info("Pathway analysis complete!")
            #
            # if patient_ids:
            #     top_genes = analyzer.top_genes(patient_ids[0], k=50)
            #     logger.info(f"Top genes for {patient_ids[0]}:\n{top_genes}")


            # === CROSS-MODAL PATHWAY ANALYSIS ===
            crossmodal_analyzer = CrossModalPathwayAnalyzer(attention_dict['gene_idx'], attention_dict['pathway_idx'])

            for pid in patient_ids:
                label = wsi_features[pid][1]
                crossmodal_analyzer.add_patient(pid, attention_dict[pid]['cross_modal_attn'], label)

            # Save individual cross-modal results
            crossmodal_output_dir = os.path.join(test_results_dir, 'patient_crossmodal_importance')
            crossmodal_analyzer.save_patient_results(crossmodal_output_dir)

            # Cross-modal class aggregation
            crossmodal_class_results = crossmodal_analyzer.class_aggregation(k=500)
            for label, df in crossmodal_class_results.items():
                df.to_csv(os.path.join(test_results_dir, f'class_{label}_crossmodal_pathways.csv'), index=False)
                logger.info(f"Class {label} top 5 cross-modal pathways:\n{df.head(5)}")

            # Cross-modal differential analysis
            crossmodal_diff_results = crossmodal_analyzer.class_differences(k=500)
            crossmodal_diff_results.to_csv(os.path.join(test_results_dir, 'crossmodal_pathway_differences.csv'),
                                           index=False)
            crossmodal_significant = crossmodal_diff_results[crossmodal_diff_results['significant'] == True]
            logger.info(f"Found {len(crossmodal_significant)} significant cross-modal differential pathways")
            logger.info(f"Top 5 cross-modal differential pathways:\n{crossmodal_diff_results.head(5)}")

            # Cross-modal class-specific drivers
            crossmodal_drivers = crossmodal_analyzer.class_specific_drivers(k=100)
            for driver_type, df in crossmodal_drivers.items():
                driver_file = os.path.join(test_results_dir, f'{driver_type}_crossmodal_pathways.csv')
                df.to_csv(driver_file, index=False)
                logger.info(f"Top 3 {driver_type} cross-modal pathways:\n{df.head(3)}")

            logger.info("Cross-modal analysis complete!")

            # Rank-based analysis
            rank_results = crossmodal_analyzer.rank_based_analysis(k=100)
            rank_results.to_csv(os.path.join(test_results_dir, 'crossmodal_pathway_ranks.csv'), index=False)
            logger.info(f"Top 5 rank-based differences:\n{rank_results.head(5)}")

            # Run consensus analysis and save CSV files
            consensus_result = crossmodal_analyzer.consensus_pathway_analysis(test_results_dir, k_per_method=100)
            logger.info(consensus_result)

            # Load and display high confidence results
            for class_label in [0, 1]:  # Adjust based on your class labels
                csv_path = os.path.join(test_results_dir, f'class_{class_label}_consensus_pathways.csv')
                if os.path.exists(csv_path):
                    consensus_df = pd.read_csv(csv_path)
                    high_conf = consensus_df[consensus_df['confidence'] == 'high']
                    logger.info(f"Class {class_label} - High Confidence Pathways ({len(high_conf)}):")
                    logger.info(high_conf['pathway'].head(10).tolist())


            patient_ids = list(test_results['metrics']['attention_dict'].keys())[-2]
            gene_idx = attention_dict['gene_idx']
            pathway_idx = attention_dict['pathway_idx']

            gene_idx_inv = {v.item(): k for k, v in gene_idx.items()}
            pathway_idx_inv = {v.item(): k for k, v in pathway_idx.items()}

            for patient_id in patient_ids:
                patient_info = attention_dict[patient_id]
                wsi_patch_names = wsi_features[patient_id][2]['filenames']
                patch_asssignment = patient_info['hard_assignments'].squeeze(0)
                patch_asssignment = [p.item() for p in patch_asssignment]
                sorted_gene_importance = patient_info['gene_pathway_attn'].sum(dim=1).sort()





