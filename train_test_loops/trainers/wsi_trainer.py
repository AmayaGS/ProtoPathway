# train_test_loops/wsi_trainer.py

import time

from pathlib import Path

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, average_precision_score)

import torch
import torch.nn.functional as F

from train_test_loops.trainers.base_trainer import BaseTrainer
from models.Prototype import ProtoMIL_V1

from utils.loss_utils import NLLSurvLoss
from utils.survival_utils import calculate_risk
from sksurv.metrics import concordance_index_censored


class WSITrainer(BaseTrainer):
    """
    Trainer for WSI data using prototype-based models.
    """

    def __init__(self, config, experiment_logger, run_type, fold_idx=None, device=None):
        super().__init__(config, experiment_logger, device)
        self.fold_idx = fold_idx
        self.model_name = config['wsi']['model']
        self.run_type = run_type

        self.is_survival = config['execution'].get('task', 'classification') == 'survival'

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


    def create_model(self):
        """Create and initialize the WSI model."""

        output_dim = self.config['survival']['survival_bins'] if self.is_survival else self.config['num_classes']

        if self.model_name == 'Prototype':

            centroid_dir = self.config['output']['data']['dir']
            dataset_name = self.config['dataset_name']
            centroid_fold = f"wsi_centroids_{dataset_name}_{self.run_type}_{self.fold_idx}.pt"
            centroid_path = Path(centroid_dir, centroid_fold)

            if centroid_path is not None:
                f = Path(centroid_path).expanduser()

            if f is not None and f.exists():
                centroids = torch.load(f, weights_only=True, map_location=self.device)
            else:
                centroids = None

            # centroids = None

            model = ProtoMIL_V1(self.config, input_dim=self.config['wsi_training']['input_dim'],
                                embedding_dim=self.config['wsi_training']['embedding_dim'],
                                num_prototypes=self.config['wsi_training']['num_prototypes'],
                                tau=self.config['wsi_training']['tau'],
                                num_classes=output_dim,
                                init_centroids=centroids)

        elif self.model_name == 'ABMIL':

            from models.ABMIL import ABMIL

            model = ABMIL(input_dim=self.config['wsi_training']['input_dim'],
                          embedding_dim=self.config['wsi_training']['embedding_dim'],
                          attention_heads=self.config['wsi_training']['attention_heads'],
                          num_classes=output_dim)

        else:
            raise ValueError(f"Unsupported WSI model: {self.model_name}")

        if self.is_survival:
            criterion = NLLSurvLoss(alpha=self.config['survival'].get('alpha', 0.5))
        else:
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


        model = model.to(self.device)

        return model, criterion, optimizer, lr_scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, aux_train_loader=None):
        """Run one ge_training epoch for WSI model."""
        model.train()
        total_loss = 0.0

        if self.is_survival:
            all_risk_scores = []
            all_survival_times = []
            all_censorships = []
        else:
            correct = 0
            total = 0

        start_time = time.time()

        for patient_ids, data_object in train_loader.items():

            if self.is_survival:
                images, targets, patient_ids, survival_time, censorship = data_object
                images, targets = images.to(self.device), targets.unsqueeze(0).to(self.device)
                survival_time, censorship = survival_time.unsqueeze(0).to(self.device), censorship.unsqueeze(0).to(self.device)
            else:
                images, targets, _ = data_object
                images, targets = images.to(self.device), targets.to(self.device)

            outputs = model(images)

            if self.is_survival:
                # Calculate loss
                loss = criterion(outputs, targets, survival_time, censorship)
                total_loss += loss.item()

                # Calculate risk scores
                risk_scores, _ = calculate_risk(outputs)
                # Store for metrics calculation
                all_risk_scores.append(risk_scores)
                all_survival_times.append(survival_time)
                all_censorships.append(censorship)
            else:
                # Classification loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                # Calculate batch accuracy
                pred = outputs.argmax(dim=1)
                batch_correct = (pred == targets).sum().item()
                batch_total = targets.size(0)
                correct += batch_correct
                total += batch_total

            # Add prototype diversity loss if configured
            if hasattr(model, 'prototype_diversity_loss'):
                diversity_loss = model.prototype_diversity_loss()
                loss = loss + diversity_loss * self.config.get('prototype_diversity_weight', 0.1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

    def validate(self, model, val_loader, criterion, wsi_val_loader=None):
        """Validate the WSI model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_patient_ids = []

        if self.is_survival:
            all_risk_scores = []
            all_survival_times = []
            all_censorships = []
        else:
            all_preds = []
            all_targets = []
            all_probs = []

        with torch.no_grad():
            for patient_ids, data_object in val_loader.items():

                if self.is_survival:
                    images, targets, patient_ids, survival_time, censorship = data_object
                    images, targets = images.to(self.device), targets.unsqueeze(0).to(self.device)
                    survival_time, censorship = survival_time.unsqueeze(0).to(self.device), censorship.unsqueeze(0).to(
                        self.device)
                else:
                    images, targets, _ = data_object
                    images, targets = images.to(self.device), targets.to(self.device)

                outputs = model(images)

                if self.is_survival:
                    # Get survival time and censorship
                    loss = criterion(outputs, targets, survival_time, censorship)
                    total_loss += loss.item()
                    risk_scores, _ = calculate_risk(outputs)
                    all_risk_scores.append(risk_scores)
                    all_survival_times.append(survival_time)
                    all_censorships.append(censorship)
                else:
                    # Classification loss
                    loss = criterion(outputs, targets)
                    pred = outputs.argmax(dim=1)
                    probs = F.softmax(outputs, dim=1)
                    batch_correct = (pred == targets).sum().item()
                    batch_total = targets.size(0)
                    correct += batch_correct
                    total += batch_total
                    total_loss += loss.item()

                    all_preds.append(pred.cpu())
                    all_targets.append(targets.cpu())
                    all_probs.append(probs.cpu())

                all_patient_ids.append(patient_ids)

        avg_loss = total_loss / len(val_loader)

        if self.is_survival:
            all_risk_scores = torch.cat(all_risk_scores, dim=0).cpu().detach().numpy()
            all_survival_times = torch.cat(all_survival_times, dim=0).cpu().detach().numpy()
            all_censorships = torch.cat(all_censorships, dim=0).cpu().detach().numpy()
            # Calculate c-index
            all_censorships = ~all_censorships.astype(bool)
            c_index = concordance_index_censored(all_censorships, all_survival_times, all_risk_scores)
            metrics = {
                'loss': total_loss / len(val_loader),
                'c_index': c_index[0],
                'all_risk_scores': all_risk_scores,
                'all_survival_times': all_survival_times,
                'all_censorships': all_censorships,
                'patient_ids': all_patient_ids
            }
        else:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_probs = torch.cat(all_probs, dim=0).numpy()

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