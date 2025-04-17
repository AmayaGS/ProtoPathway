
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from models.GeneExpressionMLP import MLPBaseline
from utils.dataset_utils import GeneExpressionDataset


def initialise_model(config):

    if config['model'] == 'MLP':
        model = MLPBaseline(
            input_size=config['input_dim'],
            hidden_size=config['hidden_dim'],
            num_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    if torch.cuda.is_available():
        model.cuda()

    return model, criterion, optimizer, lr_scheduler


def minority_sampler(dataset):

    # Get labels
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]['target'].item())

    # Count class occurrences
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count

    # Create sample weights
    weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(weights)

    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def l1_regularization(model, l1_norm):
    weights = sum(torch.abs(p).sum() for p in model.parameters())
    return weights * l1_norm

