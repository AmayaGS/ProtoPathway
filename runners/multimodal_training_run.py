# runners/multimodal_training_run.py

import os
import pickle
import h5py
import pandas as pd

from pathlib import Path

from utils.helpers import ensure_directory
from utils.model_utils import load_gene_expression_folds
from utils.model_utils import load_wsi_folds

from utils.kmeans_init import sample_embeddings, init_prototypes

from train_test_loops.trainers.multimodal_trainer import MultimodalTrainer

from utils.visualization_utils import (
    visualize_fold_results,
    visualize_aggregated_results,
    visualize_full_training_results
)


def train_multimodal_model(config, is_full_train=False, experiment_logger=None):

    logger = experiment_logger.logger
    run_type = "FT" if is_full_train else "CV"

    # Set up paths for this specific run
    results_dir = experiment_logger.log_dir
    plots_dir = experiment_logger.plots_dir
    model_dir = experiment_logger.checkpoint_dir
    metrics_dir = experiment_logger.results_dir

    # Create directories
    for directory in [results_dir, plots_dir, model_dir, metrics_dir]:
        ensure_directory(directory)

    # Store paths in config for other functions to use
    config['current_run'] = {
        'timestamp': experiment_logger.timestamp,
        'results_dir': results_dir,
        'plots_dir': plots_dir,
        'model_dir': model_dir,
        'metrics_dir': metrics_dir,
        'run_type': run_type
    }

    logger.info(f"Starting gene expression model {run_type} run")
    logger.info(f"Using experiment name: {experiment_logger.experiment_name}")
    logger.info(f"Results will be saved to {results_dir}")

    # Load the gene expression dataset
    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)

    # Load the wsi dataset
    wsi_features_path = config['output']['data']['wsi_features']
    with open(wsi_features_path, "rb") as file:
        wsi_features = pickle.load(file)

    # Load splits
    splits_dict_path = os.path.join(
        config['output']['data']['dir'],
        f"data_splits_{config['dataset_name']}.pkl"
    )

    with open(splits_dict_path, "rb") as f:
        split_dict = pickle.load(f)

    # Prepare ge_training folds
    if is_full_train:
        logger.info("Using train/test split for full ge_training")
        ge_training_folds, ge_validation_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=False, ignore_missing=True)
        wsi_training_folds, wsi_validation_folds = load_wsi_folds(wsi_features, split_dict, is_cv=False, ignore_missing=True)
        n_folds = 1
    else:
        logger.info(f"Using {len(split_dict['CV'])} cross-validation folds")
        ge_training_folds, ge_validation_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=True, ignore_missing=True)
        wsi_training_folds, wsi_validation_folds = load_wsi_folds(wsi_features, split_dict, is_cv=True, ignore_missing=True)
        n_folds = len(split_dict['CV'])

    fold_histories = []
    fold_summaries = []

    # Train on each fold
    for fold_idx, (ge_train_fold, wsi_train_fold,
                   ge_val_fold, wsi_val_fold) \
            in enumerate(zip(ge_training_folds, wsi_training_folds,
                             ge_validation_folds, wsi_validation_folds)):

        fold_name = "Full Training" if is_full_train else f"Fold {fold_idx + 1}/{n_folds}"
        logger.info(f"=== Training {fold_name} ===")

        centroid_dir = config['output']['data']['dir']
        dataset_name = config['dataset_name']
        centroid_fold = f"wsi_centroids_{dataset_name}_{run_type}_{fold_idx}.pt"
        centroid_path = os.path.join(centroid_dir, centroid_fold)

        if centroid_path is not None:
            f = Path(centroid_path).expanduser()

        if f is not None and f.exists():
            logger.info(f"Centroids already calculated")
        else:
            # load pre-computed centroids to initialize the model
            logger.info("Calculating centroids for WSI training")
            sample_wsi_features = sample_embeddings(wsi_train_fold)
            init_prototypes(sample_wsi_features,
                             n_proto=config['wsi_training']['num_prototypes'],
                             centroid_path=centroid_path)

        mm_trainer = MultimodalTrainer(
            config=config,
            experiment_logger=experiment_logger,
            fold_idx=fold_idx
        )

        model, history = mm_trainer.train(
            ge_train_fold,    # Main training data (gene expression)
            ge_val_fold,      # Main validation data (gene expression)
            wsi_train_fold,   # Auxiliary training data (WSI)
            wsi_val_fold      # Auxiliary validation data (WSI)
        )

        # Store fold results
        fold_data = {
            'fold': fold_idx,
            'history': history,
            'model_path': os.path.join(model_dir, mm_trainer.checkpoint_name)
        }

        fold_histories.append(fold_data)

        # Generate visualizations for this fold
        logger.info(f"Generating visualizations for {fold_name}")

        metric_for_best = 'acc' if config['training']['weight_type'] == 'accuracy' else 'loss'
        mode = 'max' if config['training']['weight_type'] == 'accuracy' else 'min'

        fold_summary = visualize_fold_results(
            fold_data,
            fold_idx,
            plots_dir,
            config,
            metric_for_best=metric_for_best,
            mode=mode
        )

        fold_summaries.append(fold_summary)
        logger.info(f"Completed {fold_name} training and visualization")

    # Process results depending on run type
    if is_full_train:
        # For full training, generate comprehensive visualizations
        logger.info("Generating full training visualizations")
        visualize_full_training_results(
            fold_histories[0]['history'],
            plots_dir,
            config,
            metric_for_best=metric_for_best,
            mode=mode
        )
    else:
        # For cross-validation, generate aggregated results
        logger.info("Generating aggregated cross-validation visualizations")
        visualize_aggregated_results(
            fold_summaries,
            fold_histories,
            plots_dir,
            config
        )

    # Save all fold histories for later use
    history_path = os.path.join(metrics_dir, f"{run_type}_histories.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(fold_histories, f)

    logger.info(f"Saved complete training histories to {history_path}")
    logger.info(f"{run_type.capitalize()} completed successfully!")

    return {
        'fold_histories': fold_histories,
        'fold_summaries': fold_summaries
    }







