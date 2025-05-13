# runners/gene_training_run.py

import os
import pickle
import pandas as pd

from utils.helpers import ensure_directory
from utils.model_utils import load_gene_expression_folds

from train_test_loops.trainers.gene_trainer import GeneExpressionTrainer
from utils.visualization_utils import (
    visualize_fold_results,
    visualize_aggregated_results,
    visualize_full_training_results
)


def train_gene_expression_model(config, is_full_train=False, experiment_logger=None):
    """
    Train a gene expression model using the specified configuration.

    Args:
        config: Configuration dictionary
        is_full_train: Whether to train on the full ge_training set (vs. cross-validation)
        experiment_logger: Logger instance for the experiment

    Returns:
        Dictionary with ge_training results
    """
    logger = experiment_logger.logger
    run_type = "full_train" if is_full_train else "cross_validation"

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

    # Load the dataset
    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)

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
        training_folds, validation_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=False, ignore_missing=True)
        n_folds = 1
    else:
        logger.info(f"Using {len(split_dict['CV'])} cross-validation folds")
        training_folds, validation_folds = load_gene_expression_folds(gene_expression_df, split_dict, is_cv=True, ignore_missing=True)
        n_folds = len(split_dict['CV'])

    fold_histories = []
    fold_summaries = []

    # Train on each fold
    for fold_idx, (train_fold, val_fold) in enumerate(zip(training_folds, validation_folds)):
        fold_name = "Full Training" if is_full_train else f"Fold {fold_idx + 1}/{n_folds}"
        logger.info(f"=== Training {fold_name} ===")

        # Initialize trainer
        trainer = GeneExpressionTrainer(
            config=config,
            experiment_logger=experiment_logger,
            fold_idx=fold_idx
        )

        # Train model
        model, history = trainer.train(train_fold, val_fold)

        # Store fold results
        fold_data = {
            'fold': fold_idx,
            'history': history,
            'model_path': os.path.join(model_dir, trainer.checkpoint_name)
        }

        fold_histories.append(fold_data)

        # Generate visualizations for this fold
        logger.info(f"Generating visualizations for {fold_name}")
        is_survival = config['execution'].get('task', 'classification') == 'survival'
        metric_for_best = 'c_index' if is_survival else (
            'acc' if config['training']['weight_type'] == 'accuracy' else 'loss')
        mode = 'max' if metric_for_best in ['acc', 'c_index'] else 'min'

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
        # For full ge_training, generate comprehensive visualizations
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

    logger.info(f"Saved complete ge_training histories to {history_path}")
    logger.info(f"{run_type.capitalize()} completed successfully!")

    return {
        'fold_histories': fold_histories,
        'fold_summaries': fold_summaries
    }