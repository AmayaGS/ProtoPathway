import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader as PyGDataLoader

from utils.helpers import ensure_directory
from utils.model_utils import load_ge_cv_folds, load_ge_train_test_folds, initialise_model

from utils.dataset_utils import GeneExpressionDataset
from utils.dataset_utils import build_incidence_matrix, HypergraphDataset
from train_test_loops.train_val_loop import Trainer

from utils.visualization_utils import (
    visualize_fold_results,
    visualize_aggregated_results,
    visualize_full_training_results
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(config, is_full_train=False, experiment_logger=None):

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

    logger.info(f"Starting {run_type} run")
    logger.info(f"Using experiment name: {experiment_logger.experiment_name}")
    logger.info(f"Results will be saved to {results_dir}")

    # Load the dataset
    gene_expression_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)
    ge_input_dim = gene_expression_df.shape[1]

    labels_df = pd.read_csv(
        os.path.join(config['output']['data']['dir'], f"patient_labels_{config['dataset_name']}.csv"))

    splits_dict_path = os.path.join(config['output']['data']['dir'], f"data_splits_{config['dataset_name']}.pkl")

    # Load the cross-validation splits
    with open(splits_dict_path, "rb") as f:
        split_dict = pickle.load(f)

    if config['model']['name'] == 'Hypergraph':
        data = build_incidence_matrix(config['output']['data']['final_pathways'], gene_expression_df)

    if is_full_train:
        logger.info("Using train/test split for full training")
        training_folds, validation_folds = load_ge_train_test_folds(gene_expression_df, split_dict)
        # For full training, we have just one "fold" which is the train/test split
        n_folds = 1
    else:
        logger.info(f"Using {len(split_dict['CV'])} cross-validation folds")
        training_folds, validation_folds = load_ge_cv_folds(gene_expression_df, split_dict)
        n_folds = len(split_dict['CV'])

    fold_histories = []
    fold_summaries = []

    for fold_idx, (train_fold, val_fold) in enumerate(zip(training_folds, validation_folds)):
        fold_name = "Full Training" if is_full_train else f"Fold {fold_idx + 1}/{n_folds}"
        logger.info(f"=== Training {fold_name} ===")

        if config['model']['name'] == 'MLP':
            # Create datasets for the current fold
            train_dataset = GeneExpressionDataset(config, train_fold, labels_df)
            val_dataset = GeneExpressionDataset(config, val_fold, labels_df)

            # Create data loaders
            train_loader = DataLoader(train_dataset,
                                       batch_size=config['training']['batch_size'],
                                       num_workers= config['training']['num_workers'],
                                       shuffle=True,
                                       drop_last=False)
            val_loader = DataLoader(val_dataset,
                                     batch_size=config['training']['batch_size'],
                                     num_workers=config['training']['num_workers'],
                                     shuffle=False)

        elif config['model']['name'] == 'Hypergraph':

            train_dataset = HypergraphDataset(config, train_fold, labels_df, data)
            val_dataset = HypergraphDataset(config, val_fold, labels_df, data)

            # Create data loaders
            train_loader = PyGDataLoader(train_dataset,
                                       batch_size=config['training']['batch_size'],
                                       num_workers= config['training']['num_workers'],
                                       shuffle=True,
                                       drop_last=False)

            val_loader = PyGDataLoader(val_dataset,
                                     batch_size=config['training']['batch_size'],
                                     num_workers=config['training']['num_workers'],
                                     shuffle=False)

        # Initialize model, loss function, and other components
        model, loss_function, optimizer, lr_scheduler = initialise_model(config, ge_input_dim)

        # Initialize trainer
        trainer = Trainer(experiment_logger,
                          config,
                          train_loader,
                          val_loader,
                          model,
                          loss_function,
                          optimizer,
                          lr_scheduler,
                          fold_idx,
                          device)
        # Train the model
        model, history = trainer.train() # remember to save best model

        model_path = os.path.join(experiment_logger.checkpoint_dir, trainer.checkpoint_name)

        # Store fold results
        fold_data = {
            'fold': fold_idx,
            'history': history,
            'model_path': model_path
        }

        fold_histories.append(fold_data)

        # Generate visualizations for this fold
        logger.info(f"Generating visualizations for {fold_name}")
        fold_summary = visualize_fold_results(
            fold_data,
            fold_idx,
            plots_dir,
            config,
            metric_for_best='acc' if config['training']['weight_type'] == 'accuracy' else 'loss',
            mode='max' if config['training']['weight_type'] == 'accuracy' else 'min'
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
            metric_for_best='acc' if config['training']['weight_type'] == 'accuracy' else 'loss',
            mode='max' if config['training']['weight_type'] == 'accuracy' else 'min'
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
