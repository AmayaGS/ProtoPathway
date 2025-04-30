import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

import torch

from torch_geometric.loader import DataLoader as PyGDataLoader

from utils.helpers import ensure_directory
from utils.model_utils import initialise_model

from utils.dataset_utils import build_incidence_matrix, HypergraphDataset
from train_test_loops.testing_loop import evaluate_model, save_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(config, is_continuation=False, experiment_logger=None):

    logger =  experiment_logger.logger

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

    logger.info(f"Starting model testing")
    logger.info(f"Results will be saved to {test_results_dir}")

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

    # Load the test set
    test_data = gene_expression_df.loc[split_dict["Test"]]

    test_dataset = HypergraphDataset(config, test_data, labels_df, data)
    test_loader = PyGDataLoader(
                                test_dataset,
                                batch_size=config['training']['batch_size'],
                                num_workers=config['training']['num_workers'],
                                shuffle=False
                            )

    # Load the model
    checkpoint_name = "best_fold_0.pt"
    model_path = os.path.join(model_dir, checkpoint_name)

    model, _, _, _ = initialise_model(config, ge_input_dim)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_metrics = evaluate_model(model, test_loader, config, device)

    save_metrics(test_dataset, test_metrics, test_results_dir, config, logger)

