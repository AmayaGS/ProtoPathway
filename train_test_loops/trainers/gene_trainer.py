
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, average_precision_score)

from sksurv.metrics import concordance_index_censored

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader

from train_test_loops.trainers.base_trainer import BaseTrainer
from utils.dataset_utils import GeneExpressionDataset, HypergraphDataset
from utils.dataset_utils import build_incidence_matrix
from utils.loss_utils import NLLSurvLoss
from utils.survival_utils import calculate_risk

from models.GeneExpressionMLP import MLPBaseline
from models.ProtoPathway import PathwayEmbeddingModel


class GeneExpressionTrainer(BaseTrainer):
    """
    Trainer for gene expression data using hypergraph and MLP models.
    """

    def __init__(self, config, experiment_logger, fold_idx=None, device=None):
        super().__init__(config, experiment_logger, device)

        self.config = config
        self.fold_idx = fold_idx
        self.model_name = config['gene_expression']['model']
        self.input_dim = None  # Will be set when preparing data
        self.is_survival = config['execution'].get('task', 'classification') == 'survival'


    def prepare_data(self, train_data, val_data):
        """
        Prepare data loaders for ge_training and validation.

        Args:
            train_data: Training gene expression dataframe
            val_data: Validation gene expression dataframe

        Returns:
            train_loader, val_loader
        """
        labels_path = self.config['output']['data']['filtered_labels']
        labels_df = pd.read_csv(labels_path)

        # Set input dimension for model initialization later
        self.input_dim = train_data.shape[1]

        if self.model_name == 'MLP':
            # Create datasets
            train_dataset = GeneExpressionDataset(self.config, train_data, labels_df)
            val_dataset = GeneExpressionDataset(self.config, val_data, labels_df)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=True,
                drop_last=False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=False
            )

        elif self.model_name == 'Hypergraph':
            # Build incidence matrix for the hypergraph
            data = build_incidence_matrix(
                self.config['output']['data']['final_pathways'],
                pd.concat([train_data, val_data])
            )

            train_dataset = HypergraphDataset(self.config, train_data, labels_df, data)
            val_dataset = HypergraphDataset(self.config, val_data, labels_df, data)

            # Create dataloaders
            train_loader = PyGDataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=True,
                drop_last=False
            )

            val_loader = PyGDataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=False
            )

        else:
            raise ValueError(f"Unsupported gene expression model: {self.model_name}")

        return train_loader, val_loader

    def create_model(self):
        """Create and initialize the model."""

        if self.model_name == 'MLP':
            if self.is_survival:
                model = MLPBaseline(
                    input_size=self.input_dim,
                    hidden_size=self.config['ge_training']['hidden_dim'],
                    num_classes=self.config['survival']['survival_bins'],
                    dropout_rate=self.config['ge_training']['dropout_rate']
                    )
            else:
                model = MLPBaseline(
                    input_size=self.input_dim,
                    hidden_size=self.config['ge_training']['hidden_dim'],
                    num_classes=self.config['n_classes'],
                    dropout_rate=self.config['ge_training']['dropout_rate']
                    )

        if self.model_name == 'Hypergraph':
            if self.is_survival:
                model = PathwayEmbeddingModel(self.config, in_channels=1,
                                                hidden_channels=self.config['ge_training']['hidden_dim'],
                                                out_channels=self.config['survival']['survival_bins'],
                                                num_layers=self.config['ge_training']['num_layers'],
                                                dropout=self.config['ge_training']['dropout_rate'])
            else:
                model = PathwayEmbeddingModel(self.config, in_channels=1,
                                              hidden_channels=self.config['ge_training']['hidden_dim'],
                                              out_channels=self.config['n_classes'],
                                              num_layers=self.config['ge_training']['num_layers'],
                                              dropout=self.config['ge_training']['dropout_rate'])

        if self.is_survival:
            # Create survival loss function
            criterion = NLLSurvLoss(self.config['survival']['alpha'])
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=self.config['ge_training']['learning_rate'],
                                weight_decay=self.config['ge_training']['L2_norm'])

        sched_cfg = self.config.get("scheduler", {})
        if sched_cfg.get("use", False):  # default to False if not specified
            scheduler_type = sched_cfg["type"]
            # Get other params step, gamma, etc.
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])
        else:
            lr_scheduler = None  # No scheduler used

        if torch.cuda.is_available():
            model.cuda()

        return model, criterion, optimizer, lr_scheduler


    def train_epoch(self, model, train_loader, optimizer, criterion, aux_train_loader=None):
        """Run one ge_training epoch."""

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.is_survival:
            all_risk_scores = []
            all_survival_times = []
            all_censorships = []

        start_time = time.time()

        for batch in train_loader:

            if self.model_name == 'MLP':
                patient_id = batch['patient_id']
                data, target = batch['data'].to(self.device), batch['target'].to(self.device)
                if self.is_survival:
                    survival_time = batch['survival_time'].to(self.device)
                    censorship = batch['censorship'].to(self.device)
                outputs = model(data)

            elif self.model_name == 'Hypergraph':
                batch.to(self.device)
                patient_id = batch.patient_id
                if self.is_survival:
                    target = batch.y['target']
                    survival_time = batch.y['survival_time']
                    censorship = batch.y['censorship']
                else:
                    target = batch.y

                outputs = model(batch)

            if self.is_survival:
                risk_scores, _ = calculate_risk(outputs)
                all_risk_scores.append(risk_scores)
                all_survival_times.append(survival_time)
                all_censorships.append(censorship)
                # Calculate loss
                try:
                    loss = criterion(outputs, target, survival_time, censorship)
                    total_loss += loss.item()
                except ValueError as e:
                    print("Error in calculating loss for patient:", patient_id)
                    print("Model outputs:", outputs)
                    print("Outputs shape:", outputs.shape)
                    print("Target:", target)
                    print("Target shape:", target.shape)
                    print("Survival time shape:", survival_time.shape)
                    print("Censorship shape:", censorship.shape)
                    print("GE data:", data)
                    print("GE data shape:", data.shape)
                    raise e

            else:
                loss = criterion(outputs, target)
                # Calculate batch accuracy
                pred = outputs.argmax(dim=1)
                batch_correct = (pred == target).sum().item()
                batch_total = target.size(0)
                correct += batch_correct
                total += batch_total
                total_loss += loss.item()

            # Apply L1 regularization if configured
            if self.config['ge_training']['L1_norm'] > 0:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = loss + self.config['ge_training']['L1_norm'] * l1_loss

            # Update metrics
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)

        if self.is_survival:
            all_risk_scores = torch.cat(all_risk_scores, dim=0).cpu().detach().numpy()
            all_survival_times = torch.cat(all_survival_times, dim=0).cpu().detach().numpy()
            all_censorships = torch.cat(all_censorships, dim=0).cpu().detach().numpy()
            # Calculate c-index
            all_censorships = ~all_censorships.astype(bool)
            c_index = concordance_index_censored(all_censorships, all_survival_times, all_risk_scores)
            epoch_time = time.time() - start_time
            return {
                'loss': avg_loss,
                'c_index': c_index[0],
                'time': epoch_time
            }
        else:
            avg_acc = 100. * correct / total
            epoch_time = time.time() - start_time
            return {
                'loss': avg_loss,
                'acc': avg_acc,
                'time': epoch_time
            }


    def validate(self, model, val_loader, criterion, aux_val_loader=None):
        """Validate the model."""

        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        if self.is_survival:
            all_risk_scores = []
            all_survival_times = []
            all_censorships = []
        else:
            all_preds = []
            all_targets = []
            all_probs = []

        with torch.no_grad():
            for batch in val_loader:

                if self.model_name == 'MLP':
                    data, target = batch['data'].to(self.device), batch['target'].to(self.device)
                    if self.is_survival:
                        survival_time = batch['survival_time'].to(self.device)
                        censorship = batch['censorship'].to(self.device)
                    outputs = model(data)

                elif self.model_name == 'Hypergraph':
                    batch.to(self.device)
                    if self.is_survival:
                        target = batch.y['target']
                        survival_time = batch.y['survival_time']
                        censorship = batch.y['censorship']
                    else:
                        target = batch.y

                    outputs = model(batch)

                if self.is_survival:
                    risk_scores, _ = calculate_risk(outputs)
                    all_risk_scores.append(risk_scores)
                    all_survival_times.append(survival_time)
                    all_censorships.append(censorship)
                    loss = criterion(outputs, target, survival_time, censorship)
                    total_loss += loss.item()
                else:
                    loss = criterion(outputs, target)
                    pred = outputs.argmax(dim=1)
                    probs = F.softmax(outputs, dim=1)

                    total_loss += loss.item()
                    correct += (pred == target).sum().item()
                    total += target.size(0)

                    all_preds.append(pred.cpu())
                    all_targets.append(target.cpu())
                    all_probs.append(probs.cpu())

        avg_loss = total_loss / len(val_loader)

        if self.is_survival:
            all_risk_scores = torch.cat(all_risk_scores, dim=0).cpu().detach().numpy()
            all_survival_times = torch.cat(all_survival_times, dim=0).cpu().detach().numpy()
            all_censorships = torch.cat(all_censorships, dim=0).cpu().detach().numpy()
            # Calculate c-index
            all_censorships = ~all_censorships.astype(bool)
            c_index = concordance_index_censored(all_censorships, all_survival_times, all_risk_scores)

            metrics = {
                'loss': avg_loss,
                'c_index': c_index[0],
                'all_risk_scores': all_risk_scores,
                'all_survival_times': all_survival_times,
                'all_censorships': all_censorships
            }
        else:
            # Concatenate results
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_probs = torch.cat(all_probs, dim=0).numpy()

            # Calculate metrics
            avg_acc = 100. * correct / total

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
                'all_probs': all_probs
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