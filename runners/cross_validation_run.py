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

from utils.visualization_utils import (
    create_result_summary, visualize_model_results,
    plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves
)

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

    fold_histories = []

    import pandas as pd
    import pickle
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import logging
    from datetime import datetime

    import torch
    from torch.utils.data import DataLoader

    from utils.model_utils import load_ge_cv_folds, load_ge_train_test_folds, initialise_model
    from utils.logging_utils import ExperimentLogger
    from utils.dataset_utils import GeneExpressionDataset
    from utils.helpers import ensure_directory
    from train_test_loops.train_val_loop import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('protopathway')

    def cross_validation(config, is_full_train=False):
        """
        Run cross-validation or full training based on config

        Args:
            config: Configuration dictionary
            is_full_train: If True, use train/test split instead of CV folds
        """
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_type = "full_train" if is_full_train else "cross_validation"

        # Set up directories for results
        results_dir = os.path.join(config['output_dir'], 'results', f"{run_type}_{timestamp}")
        plots_dir = os.path.join(results_dir, 'plots')
        model_dir = os.path.join(results_dir, 'models')
        metrics_dir = os.path.join(results_dir, 'metrics')

        # Create directories
        for directory in [results_dir, plots_dir, model_dir, metrics_dir]:
            ensure_directory(directory)

        # Store paths in config for other functions to use
        config['current_run'] = {
            'timestamp': timestamp,
            'results_dir': results_dir,
            'plots_dir': plots_dir,
            'model_dir': model_dir,
            'metrics_dir': metrics_dir,
            'run_type': run_type
        }

        logger.info(f"Starting {run_type} run with timestamp {timestamp}")
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

        if is_full_train or config['execution']['full_train']:
            logger.info("Using train/test split for full training")
            training_folds, validation_folds = load_ge_train_test_folds(gene_expression_df, split_dict)
        else:
            logger.info(f"Using {len(split_dict['CV'])} cross-validation folds")
            training_folds, validation_folds = load_ge_cv_folds(gene_expression_df, split_dict)

        # Create a shared experiment logger for all folds
        shared_logger = ExperimentLogger(
            config,
            experiment_name=f"{run_type}_{timestamp}",
            log_dir=results_dir
        )

        # Store results from all folds for aggregation
        all_fold_results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': [],
            'best_epochs': [],
            'confusion_matrices': []
        }

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
            model, history = trainer.train()

            # Store results for aggregation
            all_fold_results['train_loss'].append(history['train']['loss'][-1])
            all_fold_results['train_acc'].append(history['train']['acc'][-1])
            all_fold_results['val_loss'].append(history['val']['loss'][-1])
            all_fold_results['val_acc'].append(history['val']['acc'][-1])

            if 'auc' in history['val']:
                all_fold_results['val_auc'].append(history['val']['auc'][-1])

            if 'f1' in history['val']:
                all_fold_results['val_f1'].append(history['val']['f1'][-1])

            if 'confusion_matrix' in history['val']:
                all_fold_results['confusion_matrices'].append(history['val']['confusion_matrix'])

            # Store full history for this fold
            fold_histories.append({
                'fold': fold_idx,
                'history': history
            })


    print("Training complete")