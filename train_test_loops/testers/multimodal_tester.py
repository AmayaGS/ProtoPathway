# train_test_loops/testers/multimodal_tester.py
from email.policy import strict

import pandas as pd

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader as PyGDataLoader

from train_test_loops.testers.base_tester import BaseTester

from utils.dataset_utils import HypergraphDataset, build_incidence_matrix
from utils.loss_utils import NLLSurvLoss
from models.MultimodalFusionModel import ProtoPathwayFusion



class MultimodalTester(BaseTester):
    """
    Tester for multimodal models integrating gene expression and WSI data.
    Handles model evaluation for both classification and survival tasks.
    """

    def __init__(self, config, experiment_logger, fold_idx, device=None):
        """Initialize the multimodal tester."""
        super().__init__(config, experiment_logger, device)

        self.config = config

        self.fold_idx = fold_idx
        self.fusion_type = config['multimodal']['fusion_type']

        self.ge_model_name = config['gene_expression']['model']
        self.wsi_model_name = config['wsi']['model']

        self.hypergraph_data = None
        self.is_survival = config['execution'].get('task', 'classification') == 'survival'

        # Check if both modalities are enabled
        if not (config['gene_expression']['enabled'] and config['wsi']['enabled']):
            raise ValueError("Both gene expression and WSI modalities must be enabled for multimodal testing")


    def prepare_test_data(self, ge_test_data, wsi_test_data):

        labels_df = pd.read_csv(self.config['output']['data']['filtered_labels'])

        if self.ge_model_name == "Hypergraph":
            self.hypergraph_data = build_incidence_matrix(
                self.config['output']['data']['final_pathways'],
                ge_test_data
            )

            ge_test_dataset = HypergraphDataset(
                self.config, ge_test_data, labels_df, self.hypergraph_data
            )

        ge_test_patients = set(ge_test_dataset.patient_ids)
        wsi_test_patients = set(wsi_test_data.keys())
        common_test_patients = ge_test_patients.intersection(wsi_test_patients)

        # Filter the test data to only include common patients
        ge_test_dataset.patient_ids = [pid for pid in ge_test_dataset.patient_ids if pid in common_test_patients]

        ge_val_loader = PyGDataLoader(
            ge_test_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=False
        )

        wsi_test_loader = wsi_test_data

        return ge_val_loader, wsi_test_loader


    def load_model(self, model_path):
        """Load the multimodal model from the specified path."""

        if self.fusion_type == "ProtoPathway":
            model = ProtoPathwayFusion(
                config=self.config,
                centroids=None,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        if self.is_survival:
            # Create survival loss function
            criterion = NLLSurvLoss(self.config['survival']['alpha'])
        else:
            criterion = nn.CrossEntropyLoss()

        # Load state dict
        state_dict = torch.load(model_path, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)

        return model, criterion

    def _process_batch(self, model, ge_batch, wsi_data=None):

        ge_batch.to(self.device)
        patient_id = ge_batch.patient_id[0]  # Assuming batch size 1

        if self.is_survival:
            target = ge_batch.y['target']
            survival_time = ge_batch.y['survival_time']
            censorship = ge_batch.y['censorship']
        else:
            target = ge_batch.y

        # Get WSI data for this patient
        wsi_features = wsi_data[patient_id][0].to(self.device)

        # Forward pass
        outputs = model(ge_batch, wsi_features)

        if self.is_survival:
            return outputs, target, [patient_id], survival_time, censorship
        else:
            return outputs, target, [patient_id]


    def visualize_results(self, metrics):
        """
        Generate visualizations of test results.
        Placeholder for future multimodal-specific visualizations.

        Args:
            metrics: Dictionary containing evaluation metrics

        Returns:
            Dictionary of paths to visualization files (empty for now)
        """
        # Placeholder - no visualizations implemented yet
        return {}


    def evaluate_with_importance(self, model, ge_test_loader, wsi_test_loader):


        pass


    def run_testing(self, ge_test_data, wsi_test_data=None, checkpoint_path=None):
        """
        Run complete testing process for multimodal models.

        Args:
            ge_test_data: Gene expression test data
            wsi_test_data: WSI test data
            checkpoint_path: Path to model checkpoint

        Returns:
            Dictionary of test results and paths to outputs
        """
        # Use the base class run_testing method with multimodal data
        return super().run_testing(ge_test_data, wsi_test_data, checkpoint_path)
