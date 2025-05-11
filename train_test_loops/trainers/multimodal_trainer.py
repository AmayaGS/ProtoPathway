# train_test_loops/trainers/multimodal_trainer.py

import os
import time
import torch
import torch.nn.functional as F

import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, average_precision_score)

from torch_geometric.loader import DataLoader as PyGDataLoader

from train_test_loops.trainers.base_trainer import BaseTrainer

from utils.dataset_utils import HypergraphDataset
from utils.dataset_utils import build_incidence_matrix
from utils.model_utils import l1_regularization


class MultimodalTrainer(BaseTrainer):
    """
    Trainer for multimodal data integrating gene expression and WSI data.
    """

    def __init__(
            self,
            config,
            experiment_logger,
            fold_idx=None,
            device=None
    ):
        """
        Initialize the multimodal trainer.

        Args:
            config: Dictionary containing configuration parameters
            experiment_logger: Logger instance for the experiment
            fold_idx: Current fold index (for cross-validation)
            device: Computation device (CPU/GPU)
        """
        super().__init__(config, experiment_logger, device)
        self.fold_idx = fold_idx
        self.fusion_type = config['multimodal']['fusion_type']

        self.ge_model_name = config['gene_expression']['model']
        self.wsi_model_name = config['wsi']['model']

        # Modality-specific parameters
        self.gene_expr_enabled = config['gene_expression']['enabled']
        self.wsi_enabled = config['wsi']['enabled']

        # Ensure both modalities are enabled for multimodal training
        if not (self.gene_expr_enabled and self.wsi_enabled):
            raise ValueError("Both gene expression and WSI modalities must be enabled for multimodal training")

        # Track input dimensions for model creation
        self.gene_expr_dim = None
        self.wsi_feature_dim = None
        self.hypergraph_data = None

    def prepare_data(self, train_data, val_data, aux_train_data=None, aux_val_data=None):
        """
        Prepare data loaders for training and validation.

        Args:
            train_data: Gene expression training data
            val_data: Gene expression validation data
            aux_train_data: WSI training data
            aux_val_data: WSI validation data

        Returns:
            Tuple of (ge_train_loader, ge_val_loader, wsi_train_loader, wsi_val_loader)
        """

        # For multimodal training, we need both gene expression and WSI data
        if aux_train_data is None or aux_val_data is None:
            self.logger.logger.warning("Multimodal training requires both gene expression and WSI data")
            # Use the provided data as gene expression, and no WSI data
            ge_train_data, ge_val_data = train_data, val_data
            wsi_train_data, wsi_val_data = {}, {}
        else:
            # Use the provided data as expected
            ge_train_data, ge_val_data = train_data, val_data
            wsi_train_data, wsi_val_data = aux_train_data, aux_val_data

        labels_df = pd.read_csv(
            os.path.join(self.config['output']['data']['dir'],
                         f"patient_labels_{self.config['dataset_name']}.csv"))

        if self.ge_model_name == 'Hypergraph':
            # Build hypergraph representation
            data = build_incidence_matrix(
                self.config['output']['data']['final_pathways'],
                pd.concat([ge_train_data, ge_val_data])
            )
            self.hypergraph_data = data

            # Create datasets
            ge_train_dataset = HypergraphDataset(self.config, ge_train_data, labels_df, data)
            ge_val_dataset = HypergraphDataset(self.config, ge_val_data, labels_df, data)

            # Create dataloaders
            ge_train_loader = PyGDataLoader(
                ge_train_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=True,
                drop_last=False
            )

            ge_val_loader = PyGDataLoader(
                ge_val_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=False
            )

        # For WSI data, we just use the dictionaries directly
        wsi_train_loader = wsi_train_data
        wsi_val_loader = wsi_val_data

        return ge_train_loader, ge_val_loader, wsi_train_loader, wsi_val_loader

    # def prepare_data(self, ge_train_data, ge_val_data,
    #                  wsi_train_data, wsi_val_data):
    #     """
    #     Prepare data loaders for ge_training and validation.
    #
    #     Args:
    #         train_data: Dictionary with 'gene_expression' and 'wsi' data
    #         val_data: Dictionary with 'gene_expression' and 'wsi' data
    #
    #     Returns:
    #         train_loader, val_loader
    #     """
    #
    #     if self.ge_model_name == 'Hypergraph':
    #
    #         labels_df = pd.read_csv(
    #             os.path.join(self.config['output']['data']['dir'],
    #                          f"patient_labels_{self.config['dataset_name']}.csv"))
    #
    #         data = build_incidence_matrix(
    #             self.config['output']['data']['final_pathways'],
    #             pd.concat([ge_train_data, ge_val_data])
    #         )
    #
    #         ge_train_dataset = HypergraphDataset(self.config, ge_train_data, labels_df, data)
    #         ge_val_dataset = HypergraphDataset(self.config, ge_val_data, labels_df, data)
    #
    #         # Create dataloaders
    #         ge_train_loader = PyGDataLoader(
    #             ge_train_dataset,
    #             batch_size=self.config['ge_training']['batch_size'],
    #             num_workers=self.config['ge_training']['num_workers'],
    #             shuffle=True,
    #             drop_last=False
    #         )
    #
    #         ge_val_loader = PyGDataLoader(
    #             ge_val_dataset,
    #             batch_size=self.config['ge_training']['batch_size'],
    #             num_workers=self.config['ge_training']['num_workers'],
    #             shuffle=False
    #         )
    #
    #         if self.wsi_model_name == 'Prototype':
    #             # WSI data is already in the correct format
    #             wsi_train_loader = wsi_train_data
    #             wsi_val_loader = wsi_val_data
    #
    #         return ge_train_loader, ge_val_loader, wsi_train_loader, wsi_val_loader


    def create_model(self):
        """
        Create and initialize the multimodal model.

        Returns:
            model, criterion, optimizer, lr_scheduler
        """
        from models.MultimodalFusionModel import ProtoPathwayFusion

        # Create multimodal fusion model
        model = ProtoPathwayFusion(
            config=self.config,
            device=self.device
        )

        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['mm_training']['learning_rate'],
            weight_decay=self.config['mm_training']['L2_norm']
        )

        # Configure learning rate scheduler if needed
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
                    mode='min' if self.weight_type == 'loss' else 'max',
                    patience=self.config['scheduler']['patience'],
                    factor=self.config['scheduler']['gamma'],
                    min_lr=self.config['scheduler']['min_lr']
                )

        # Move model to device
        model = model.to(self.device)

        return model, criterion, optimizer, lr_scheduler

    def train_epoch(self, model, ge_train_loader, optimizer, criterion, wsi_train_loader=None):
        """
        Run one ge_training epoch.

        Args:
            model: The model to train
            ge_train_loader: DataLoader for gene expression training data
            optimizer: The optimizer
            criterion: Loss function
            wsi_train_loader: DataLoader for WSI training data

        Returns:
            Dictionary of metrics for this epoch
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(ge_train_loader):

            batch.to(self.device)
            patient_id = batch.patient_id
            target = batch.y
            ge_data = batch

            wsi_data = wsi_train_loader[patient_id[0]]
            wsi_emb = wsi_data[0]
            wsi_emb = wsi_emb.to(self.device)

            optimizer.zero_grad()
            outputs = model(ge_data, wsi_emb)

            # Calculate loss
            loss = criterion(outputs, target)

            # Add L1 regularization if configured
            if self.config['mm_training']['L1_norm'] > 0:
                l1_loss = l1_regularization(model, self.config['mm_training']['L1_norm'])
                loss = loss + l1_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            pred = outputs.argmax(dim=1)
            batch_correct = (pred == target).sum().item()
            batch_total = target.size(0)

            # Update metrics
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

        # Calculate epoch metrics
        avg_loss = total_loss / len(ge_train_loader)
        avg_acc = 100. * correct / total if total > 0 else 0
        epoch_time = time.time() - start_time

        # Return metrics
        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'time': epoch_time
        }

    def validate(self, model, ge_val_loader, criterion, wsi_val_loader=None):
        """
        Validate the model.

        Args:
            model: The model to validate
            ge_val_loader: DataLoader for gene expression validation data
            criterion: Loss function
            wsi_val_loader: DataLoader for WSI validation data

        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []
        all_patient_ids = []

        # Disable gradient calculation for validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(ge_val_loader):
                batch.to(self.device)
                patient_id = batch.patient_id
                target = batch.y
                ge_data = batch

                wsi_data = wsi_val_loader[patient_id]

                outputs = model(ge_data, wsi_data)

                # Calculate loss
                loss = criterion(outputs, target)
                pred = outputs.argmax(dim=1)
                probs = F.softmax(outputs, dim=1)

                # Update metrics
                total_loss += loss.item()
                correct += (pred == target).sum().item()
                total += target.size(0)

                # Store predictions and targets
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_probs.append(probs.cpu())
                all_patient_ids.extend(patient_id)

        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        # Calculate metrics
        avg_loss = total_loss / len(ge_val_loader)
        avg_acc = 100. * correct / total if total > 0 else 0

        # Compile all metrics
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

    # def get_embedding_visualizations(self, model, val_loader):
    #     """
    #     Generate modality-specific embeddings for visualization.
    #
    #     Args:
    #         model: Trained multimodal model
    #         val_loader: Validation data loader
    #
    #     Returns:
    #         Dictionary with embeddings and metadata
    #     """
    #     model.eval()
    #
    #     patient_ids = []
    #     labels = []
    #     gene_embeddings = []
    #     wsi_embeddings = []
    #     fused_embeddings = []
    #
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             # Extract data
    #             gene_expr_data = batch['gene_expression'].to(self.device) if self.gene_expr_enabled else None
    #             wsi_data = batch['wsi'].to(self.device) if self.wsi_enabled else None
    #             target = batch['target'].to(self.device)
    #             patient_id = batch['patient_id']
    #
    #             hypergraph_data = None
    #             if self.hypergraph_data is not None and 'hypergraph' in batch:
    #                 hypergraph_data = batch['hypergraph'].to(self.device)
    #
    #             # Get embeddings from model
    #             if hasattr(model, 'get_embeddings'):
    #                 gene_emb, wsi_emb, fused_emb = model.get_embeddings(
    #                     gene_expr_data, wsi_data, hypergraph_data
    #                 )
    #
    #                 # Store embeddings and metadata
    #                 gene_embeddings.append(gene_emb.cpu())
    #                 wsi_embeddings.append(wsi_emb.cpu())
    #                 fused_embeddings.append(fused_emb.cpu())
    #                 patient_ids.extend(patient_id)
    #                 labels.append(target.cpu())
    #
    #     # Concatenate results
    #     if gene_embeddings and wsi_embeddings and fused_embeddings:
    #         gene_embeddings = torch.cat(gene_embeddings, dim=0).numpy()
    #         wsi_embeddings = torch.cat(wsi_embeddings, dim=0).numpy()
    #         fused_embeddings = torch.cat(fused_embeddings, dim=0).numpy()
    #         labels = torch.cat(labels, dim=0).numpy()
    #
    #         return {
    #             'patient_ids': patient_ids,
    #             'labels': labels,
    #             'gene_embeddings': gene_embeddings,
    #             'wsi_embeddings': wsi_embeddings,
    #             'fused_embeddings': fused_embeddings
    #         }
    #
    #     return None