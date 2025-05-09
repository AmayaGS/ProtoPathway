# train_test_loops/wsi_trainer.py

import os
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, average_precision_score)

import torch
import torch.nn.functional as F

from train_test_loops.trainers.base_trainer import BaseTrainer
from models.Prototype import ProtoMIL_V0


class WSITrainer(BaseTrainer):
    """
    Trainer for WSI data using prototype-based models.
    """

    def __init__(self, config, experiment_logger, fold_idx=None, device=None):
        super().__init__(config, experiment_logger, device)
        self.fold_idx = fold_idx
        self.model_name = config['wsi']['model']

    def prepare_data(self, train_data, val_data):
        """
        Prepare data loaders for ge_training and validation.

        Args:
            train_data: Training WSI data dictionary
            val_data: Validation WSI data dictionary

        Returns:
            train_loader, val_loader
        """

        return train_data, val_data
        # labels_df = pd.read_csv(
        #     os.path.join(self.config['output']['data']['dir'],
        #                  f"patient_labels_{self.config['dataset_name']}.csv"))
        #
        # # For WSI data, we need to create patch-based datasets
        # if self.model_name == 'ProtoNet':
        #     # Create patch datasets
        #     train_dataset = WSIPatchDataset(
        #         self.config,
        #         train_data,
        #         labels_df,
        #         transform=get_transform(self.config, is_training=True)
        #     )
        #
        #     val_dataset = WSIPatchDataset(
        #         self.config,
        #         val_data,
        #         labels_df,
        #         transform=get_transform(self.config, is_training=False)
        #     )
        #
        #     # Create dataloaders
        #     train_loader = DataLoader(
        #         train_dataset,
        #         batch_size=self.config['ge_training']['batch_size'],
        #         num_workers=self.config['ge_training']['num_workers'],
        #         shuffle=True,
        #         drop_last=False
        #     )
        #
        #     val_loader = DataLoader(
        #         val_dataset,
        #         batch_size=self.config['ge_training']['batch_size'],
        #         num_workers=self.config['ge_training']['num_workers'],
        #         shuffle=False
        #     )
        #
        # else:
        #     raise ValueError(f"Unsupported WSI model: {self.model_name}")
        #
        # return train_loader, val_loader

    def create_model(self):
        """Create and initialize the WSI model."""
        if self.model_name == 'Prototype':
            model = ProtoMIL_V0(
                input_dim = self.config['wsi_training']['input_dim'],
                num_prototypes=self.config['wsi_training']['num_prototypes'],
                tau=self.config['wsi_training']['tau'],
                num_classes = self.config['n_classes']
            )

            # Define criterion, optimizer and scheduler
            criterion = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['wsi_training']['learning_rate'],
                weight_decay=self.config['wsi_training']['L2_norm']
            )

            # Configure scheduler if needed
            lr_scheduler = None
            if self.config['scheduler']['use']:
                if self.config['scheduler']['type'] == 'step':
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=self.config['scheduler']['step'],
                        gamma=self.config['scheduler']['gamma']
                    )
                elif self.config['scheduler']['type'] == 'plateau':
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min' if self.config['wsi_training']['weight_type'] == 'loss' else 'max',
                        patience=self.config['scheduler']['patience'],
                        factor=self.config['scheduler']['gamma']
                    )

        else:
            raise ValueError(f"Unsupported WSI model: {self.model_name}")

        model = model.to(self.device)

        return model, criterion, optimizer, lr_scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Run one ge_training epoch for WSI model."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for patient_ids, data_object in train_loader.items():
            images, targets, _ = data_object
            images, targets = images.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs, prototype_distances = model(images)

            # Classification loss
            loss = criterion(outputs, targets)

            # Add prototype diversity loss if configured
            if hasattr(model, 'prototype_diversity_loss'):
                diversity_loss = model.prototype_diversity_loss()
                loss = loss + diversity_loss * self.config.get('prototype_diversity_weight', 0.1)

            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            pred = outputs.argmax(dim=1)
            batch_correct = (pred == targets).sum().item()
            batch_total = targets.size(0)

            # Update metrics
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        epoch_time = time.time() - start_time

        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'time': epoch_time
        }

    def validate(self, model, val_loader, criterion):
        """Validate the WSI model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []
        all_patient_ids = []

        with torch.no_grad():
            for patient_ids, data_object in val_loader.items():
                images, targets, _ = data_object
                images, targets = images.to(self.device), targets.to(self.device)

                outputs, prototype_distances = model(images)
                loss = criterion(outputs, targets)

                probs = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)

                total_loss += loss.item()
                correct += (pred == targets).sum().item()
                total += targets.size(0)

                all_preds.append(pred.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())
                all_patient_ids.extend(patient_ids)

        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        # Calculate metrics
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0

        metrics = {
            'loss': avg_loss,
            'acc': avg_acc,
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds, zero_division=0),
            'all_labels': all_targets,
            'all_preds': all_preds,
            'all_probs': all_probs,
            'patient_ids': all_patient_ids
        }

        # Calculate AUC if binary classification
        if all_probs.shape[1] == 2:
            metrics['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            n_classes = self.config['num_classes']
            binary_labels = label_binarize(all_targets, classes=list(range(n_classes)))
            metrics['auc'] = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')
            metrics['precision'] = average_precision_score(
                binary_labels, all_probs, average='macro'
            )

        return metrics