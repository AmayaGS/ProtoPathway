import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from typing import Dict, List, Tuple, Optional, Any, Callable

from utils.logging_utils import ExperimentLogger
from utils.model_utils import initialise_model, minority_sampler


class Trainer:

    def __init__(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            config: Dict[str, Any],
            logger: ExperimentLogger,
            device: torch.device = None,
            fold: Optional[int] = None
    ):

        model, criterion, optimizer, lr_scheduler = initialise_model(config)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.fold = fold

        self.num_epochs = config['training']['num_epochs']

        self.weight_type = config['training']['weight_type']
        self.metric_to_track = 'loss' if self.weight_type == 'loss' else 'acc' if self.weight_type == 'accuracy' else 'auc'
        self.mode = 'min' if self.weight_type == 'loss' else 'max'

        self.best_val_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.current_epoch = 0

        self.checkpoint_name = f"best_fold_{fold}.pt" if fold is not None else "best_model.pt"

        self._print_training_info()


    def _print_training_info(self):

        fold_str = f"Fold {self.fold}" if self.fold is not None else "Full dataset"
        print(f"Training {fold_str} on {self.device}")
        print(f"Model: {self.config['model']['name']}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}, LR={self.lr}, Weight Decay={self.weight_decay}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Tracking {self.metric_to_track} ({self.mode}) for best model\n")


    def train_epoch(self):

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        # Track individual batch metrics
        batch_metrics = {
            'loss': [],
            'acc': []
        }

        for batch_idx, batch in enumerate(self.train_loader):
            data, target = batch['data'].to(self.device), batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            # Calculate batch accuracy
            pred = outputs.argmax(dim=1)
            batch_correct = (pred == target).sum().item()
            batch_total = target.size(0)

            # Update metrics
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

            # Track batch metrics
            batch_metrics['loss'].append(loss.item())
            batch_metrics['acc'].append(100. * batch_correct / batch_total)

            # Log batch metrics occasionally
            if batch_idx % 10 == 0:
                batch_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.logger.log_metric('batch_loss', loss.item(), batch_step, 'train')
                self.logger.log_metric('batch_acc', 100. * batch_correct / batch_total, batch_step, 'train')

                # Print progress
                print(f'Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100. * batch_correct / batch_total:.2f}%')

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        epoch_time = time.time() - start_time

        # Log epoch metrics
        metrics = {
            'loss': avg_loss,
            'acc': avg_acc,
            'time': epoch_time
        }

        return metrics


    def validate(self):

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in self.val_loader:
                data, target = batch['data'].to(self.device), batch['target'].to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, target)
                pred = outputs.argmax(dim=1)
                probs = torch.nn.functional.softmax(outputs, dim=1)

                total_loss += loss.item()
                correct += (pred == target).sum().item()
                total += target.size(0)

                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total

        metrics = {
            'loss': avg_loss,
            'acc': avg_acc,
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        }

        # Calculate AUC if binary classification
        if all_probs.shape[1] == 2:
            from sklearn.metrics import roc_auc_score
            metrics['auc'] = roc_auc_score(all_targets, all_probs[:, 1])

        return metrics


    def train(self):

        history = {
            'train': {
                'loss': [],
                'acc': []
            },
            'val': {
                'loss': [],
                'acc': []
            }
        }

        # Training loop
        for epoch in range(1, self.num_epochs + 1):

            self.current_epoch = epoch
            self.logger.start_timer(f"epoch_{epoch}")
            train_metrics = self.train_epoch()
            self.logger.log_metrics(train_metrics, epoch, 'train')

            # Validate
            val_metrics = self.validate()
            self.logger.log_metrics(val_metrics, epoch, 'val')

            # Update training history
            for metric, value in train_metrics.items():
                if metric in history['train']:
                    history['train'][metric].append(value)

            for metric, value in val_metrics.items():
                if metric in history['val']:
                    history['val'][metric].append(value)

            # Update learning rate scheduler if configured
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics[self.metric_to_track] * (-1 if self.mode == 'max' else 1))
                else:
                    self.lr_scheduler.step()

            # Check if this is the best model so far
            current_metric = val_metrics[self.metric_to_track]
            is_best = (self.mode == 'min' and current_metric < self.best_val_metric) or \
                      (self.mode == 'max' and current_metric > self.best_val_metric)

            if is_best:
                self.best_val_metric = current_metric

                # Save best model checkpoint
                if self.config['training']['checkpoint']:
                    checkpoint_path = self.logger.save_checkpoint(
                        self.model, self.optimizer, epoch,
                        {**val_metrics, **train_metrics},
                        self.checkpoint_name
                    )
                    print(f"Saved best model to {checkpoint_path}")

            # Stop epoch timer
            epoch_time = self.logger.stop_timer(f"epoch_{epoch}")

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.2f}%")
            if 'f1' in val_metrics:
                print(f"Val F1: {val_metrics['f1']:.4f}, Val Precision: {val_metrics['precision']:.4f}, "
                      f"Val Recall: {val_metrics['recall']:.4f}")
            if 'auc' in val_metrics:
                print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Best val {self.metric_to_track}: {self.best_val_metric:.4f}\n" + "-" * 50)

        # Load best model if requested
        if self.config['training']['checkpoint']:
            best_path = os.path.join(self.logger.checkpoint_dir, self.checkpoint_name)
            if os.path.exists(best_path):
                print(f"Loading best model from {best_path}")
                self.model, checkpoint = self.logger.load_checkpoint(
                    self.model, self.optimizer, best_path, self.device
                )
                best_epoch = checkpoint['epoch']
                print(f"Loaded best model from epoch {best_epoch}")

        return self.model, history





























def train_val_loop(config, train_loader, val_loader, logger, fold_idx, checkpoint=True, checkpoint_path=None):


    model, criterion, optimizer, lr_scheduler = initialise_model(config)

    num_epochs = config['training']['num_epochs']

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Best validation accuracy
    best_val_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        for batch in train_loader:
            # Get data
            data = batch['data']
            targets = batch['target']

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []

