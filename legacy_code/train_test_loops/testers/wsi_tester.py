# train_test_loops/testers/wsi_tester.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch.utils.data import DataLoader

from legacy_code.train_test_loops.testers.base_tester import BaseTester
from utils.wsi_dataset import WSIDataset, WSIPatchDataset
from models.WSIModels.prototype_model import PrototypeModel


class WSITester(BaseTester):
    """
    Tester for WSI prototype models.
    Handles model evaluation and prototype visualization.
    """

    def __init__(self, config, experiment_logger, device=None):
        """Initialize the WSI tester."""
        super().__init__(config, experiment_logger, device)
        self.model_name = config['wsi']['model']

    def prepare_test_data(self, test_data):
        """
        Prepare test data loader.

        Args:
            test_data: WSI test data

        Returns:
            test_loader
        """
        # Load patient labels
        labels_df = pd.read_csv(
            os.path.join(self.config['output']['data']['dir'],
                         f"patient_labels_{self.config['dataset_name']}.csv"))

        # For WSI data, we need to create patch-based dataset
        from utils.wsi_dataset import get_transform

        test_dataset = WSIPatchDataset(
            self.config,
            test_data,
            labels_df,
            transform=get_transform(self.config, is_training=False)
        )

        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['ge_training']['batch_size'],
            num_workers=self.config['ge_training']['num_workers'],
            shuffle=False
        )

        return test_loader

    def load_model(self, checkpoint_path):
        """
        Load a trained WSI model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model
        """
        if self.model_name == 'ProtoNet':
            # Initialize the model
            model = PrototypeModel(
                num_classes=self.config['num_classes'],
                num_prototypes=self.config['wsi']['prototype']['num_prototypes'],
                backbone=self.config['wsi']['prototype']['backbone'],
                pretrained=False  # No need for pretrained weights when loading checkpoint
            )

            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)

        else:
            raise ValueError(f"Unsupported WSI model: {self.model_name}")

        model = model.to(self.device)
        model.eval()

        return model

    def _process_batch(self, model, batch):
        """
        Process a batch of WSI data for evaluation.

        Args:
            model: The model being evaluated
            batch: A batch of data

        Returns:
            Tuple of (outputs, targets, patient_ids)
        """
        images, targets, patient_ids, *_ = batch
        images, targets = images.to(self.device), targets.to(self.device)

        if self.model_name == 'ProtoNet':
            outputs, _ = model(images)
        else:
            outputs = model(images)

        return outputs, targets, patient_ids

    def visualize_prototypes(self, model, test_loader, num_prototypes=10):
        """
        Visualize WSI prototypes and their activations.

        Args:
            model: Trained prototype model
            test_loader: Test data loader
            num_prototypes: Number of prototypes to visualize

        Returns:
            Dictionary of visualization paths
        """
        if self.model_name != 'ProtoNet' or not hasattr(model, 'get_prototype_activations'):
            self.logger.logger.warning("Prototype visualization only available for ProtoNet model")
            return {}

        # Create directory for prototype visualizations
        proto_dir = os.path.join(self.test_results_dir, 'prototypes')
        os.makedirs(proto_dir, exist_ok=True)

        visualization_paths = {}

        # Get a batch of test images
        images, targets, patient_ids, *_ = next(iter(test_loader))
        images, targets = images.to(self.device), targets.to(self.device)

        # Get prototype activations
        _, prototype_activations = model(images, return_activations=True)

        # For each prototype, find the image with highest activation
        num_prototypes = min(num_prototypes, prototype_activations.shape[1])

        for p in range(num_prototypes):
            activations = prototype_activations[:, p].detach().cpu().numpy()
            max_activation_idx = np.argmax(activations)

            # Get the image with highest activation
            max_image = images[max_activation_idx].cpu().numpy().transpose(1, 2, 0)

            # Normalize image for visualization
            max_image = (max_image - max_image.min()) / (max_image.max() - max_image.min())

            # Plot image and activation
            plt.figure(figsize=(12, 6))

            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(max_image)
            plt.title(f"Image with highest activation")
            plt.axis('off')

            # Plot activation map (if available)
            if hasattr(model, 'get_prototype_activation_map'):
                activation_map = model.get_prototype_activation_map(
                    images[max_activation_idx:max_activation_idx + 1], p
                )

                plt.subplot(1, 2, 2)
                plt.imshow(activation_map.squeeze().cpu().numpy(), cmap='jet')
                plt.title(f"Activation Map")
                plt.axis('off')

            # Save visualization
            proto_path = os.path.join(proto_dir, f'prototype_{p}.png')
            plt.savefig(proto_path, bbox_inches='tight')
            plt.close()

            visualization_paths[f'prototype_{p}'] = proto_path

        # Create prototype importance visualization
        if hasattr(model, 'get_prototype_importance'):
            importance = model.get_prototype_importance().detach().cpu().numpy()

            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance)), importance)
            plt.xlabel('Prototype Index')
            plt.ylabel('Importance')
            plt.title('Prototype Importance')

            importance_path = os.path.join(proto_dir, 'prototype_importance.png')
            plt.savefig(importance_path, bbox_inches='tight')
            plt.close()

            visualization_paths['prototype_importance'] = importance_path

        return visualization_paths

    def visualize_results(self, metrics):
        """
        Generate visualizations of WSI test results.

        Args:
            metrics: Dictionary containing evaluation metrics

        Returns:
            Dictionary of paths to visualization files
        """
        viz_dir = os.path.join(self.test_results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        visualization_paths = {}

        # 1. Confusion matrix
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.label_dict.get(str(i), f"Class_{i}") for i in range(len(self.label_dict))],
            yticklabels=[self.label_dict.get(str(i), f"Class_{i}") for i in range(len(self.label_dict))]
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        visualization_paths['confusion_matrix'] = cm_path

        # 2. ROC curve for binary classification
        if metrics['all_probs'].shape[1] == 2:
            fpr, tpr, _ = roc_curve(metrics['all_targets'], metrics['all_probs'][:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            roc_path = os.path.join(viz_dir, 'roc_curve.png')
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            visualization_paths['roc_curve'] = roc_path

            # 3. Precision-Recall curve
            precision, recall, _ = precision_recall_curve(
                metrics['all_targets'], metrics['all_probs'][:, 1]
            )
            pr_auc = auc(recall, precision)

            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")

            pr_path = os.path.join(viz_dir, 'pr_curve.png')
            plt.savefig(pr_path, bbox_inches='tight')
            plt.close()
            visualization_paths['pr_curve'] = pr_path

        return visualization_paths

    def run_testing(self, test_data, checkpoint_path=None):
        """
        Run complete testing process for WSI models.

        Args:
            test_data: WSI test data
            checkpoint_path: Path to model checkpoint (optional)

        Returns:
            Dictionary of test results and paths to outputs
        """
        # Call parent method for standard evaluation
        results = super().run_testing(test_data, checkpoint_path)

        # Add prototype visualizations if applicable
        if self.model_name == 'ProtoNet':
            test_loader = self.prepare_test_data(test_data)
            prototype_paths = self.visualize_prototypes(
                results['model'],
                test_loader,
                num_prototypes=min(10, self.config['wsi']['prototype']['num_prototypes'])
            )

            # Add prototype visualization paths
            if prototype_paths:
                if 'visualizations' not in results['output_paths']:
                    results['output_paths']['visualizations'] = {}

                results['output_paths']['visualizations'].update(prototype_paths)

        return results

