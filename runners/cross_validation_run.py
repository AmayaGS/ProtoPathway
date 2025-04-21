import pandas as pd
import pickle
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.helpers import ensure_directory
from utils.model_utils import load_ge_cv_folds, load_ge_train_test_folds, initialise_model

from utils.dataset_utils import GeneExpressionDataset
from train_test_loops.train_val_loop import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validation(config, is_full_train=False, experiment_logger=None):


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
    gene_expression_df = pd.read_csv(os.path.join(config['output']['data']['dir'],
                                                  f"shared_subset_gene_expression_{config['dataset_name']}.csv"),
                                                   index_col=0)
    ge_input_dim = gene_expression_df.shape[1]
    labels_df = pd.read_csv(
        os.path.join(config['output']['data']['dir'], f"shared_patient_labels_{config['dataset_name']}.csv"))

    splits_dict_path = os.path.join(config['output']['data']['dir'], f"data_splits_{config['dataset_name']}.pkl")

    # Load the cross-validation splits
    with open(splits_dict_path, "rb") as f:
        split_dict = pickle.load(f)

    if config['execution']['cross_validation']:
        training_folds, validation_folds = load_ge_cv_folds(gene_expression_df, split_dict)
    elif config['execution']['full_train']:
        training_folds, validation_folds = load_ge_train_test_folds(gene_expression_df, split_dict)

    for fold_idx, (train_fold, val_fold) in enumerate(zip(training_folds, validation_folds)):
        logger.info(f"Training on Fold {fold_idx + 1}")

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
        trainer.train()

    print("Training complete")