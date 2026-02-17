# train_test_loops/testers/gene_tester.py

import os
import torch

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from torch.utils.data import DataLoader

from legacy_code.train_test_loops.testers.base_tester import BaseTester
from legacy_code.utils.dataset_utils import GeneExpressionDataset, HypergraphDataset, build_incidence_matrix
from legacy_code.models.ProtoPathway import PathwayEmbeddingModel
from legacy_code.models.GeneExpressionMLP import MLPBaseline


class GeneExpressionTester(BaseTester):
    """
    Tester for gene expression models.
    Handles model evaluation and biomarker analysis.
    """

    def __init__(self, config, experiment_logger, is_continuation, device=None):
        """Initialize the gene expression tester."""
        super().__init__(config, experiment_logger, device)
        self.model_name = config['gene_expression']['model']
        self.hypergraph_data = None

    def prepare_test_data(self, test_data):
        """
        Prepare test data loader.

        Args:
            test_data: Gene expression test data

        Returns:
            test_loader
        """
        # Load patient labels
        labels_df = pd.read_csv(
            os.path.join(self.config['output']['data']['dir'],
                         f"patient_labels_{self.config['dataset_name']}.csv"))

        if self.model_name == 'MLP':
            # Create dataset
            test_dataset = GeneExpressionDataset(self.config, test_data, labels_df)

            # Create dataloader
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['ge_training']['batch_size'],
                num_workers=self.config['ge_training']['num_workers'],
                shuffle=False
            )

        elif self.model_name == 'Hypergraph':
            # Build incidence matrix for hypergraph
            self.hypergraph_data = build_incidence_matrix(
                self.config['output']['data']['final_pathways'],
                test_data
            )

            # Create dataset
            test_dataset = HypergraphDataset(
                self.config, test_data, labels_df, self.hypergraph_data
            )

            # Create dataloader
            from torch_geometric.loader import DataLoader as PyGDataLoader
            test_loader = PyGDataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                shuffle=False
            )

        else:
            raise ValueError(f"Unsupported gene expression model: {self.model_name}")

        return test_loader

    def load_model(self, checkpoint_path):
        """
        Load a trained gene expression model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model
        """
        if self.model_name == 'Hypergraph':
            # Initialize the model
            model = PathwayEmbeddingModel(self.config, in_channels=1,
                                          hidden_channels=self.config['ge_training']['hidden_dim'],
                                          out_channels=self.config['n_classes'],
                                          num_layers=self.config['ge_training']['num_layers'],
                                          dropout=self.config['ge_training']['dropout_rate'],
                                          gene_names=self.hypergraph_data[
                                              'gene_names'] if self.hypergraph_data else None,
                                          pathway_names=self.hypergraph_data[
                                              'pathway_names'] if self.hypergraph_data else None)

            # Load state dict
            state_dict = torch.load(checkpoint_path)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)

        else:  # MLP or other models
            input_dim = 9425
            model = MLPBaseline(
                input_size=input_dim,
                hidden_size=self.config['ge_training']['hidden_dim'],
                num_classes=self.config['num_classes'],
                dropout_rate=self.config['ge_training']['dropout_rate']
            )

            # Load state dict
            state_dict = torch.load(checkpoint_path, weights_only=True)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'], strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)

        model = model.to(self.device)
        model.eval()

        return model

    def _process_batch(self, model, batch):
        """
        Process a batch of gene expression data for evaluation.

        Args:
            model: The model being evaluated
            batch: A batch of data

        Returns:
            Tuple of (outputs, targets, patient_ids)
        """
        if self.model_name == 'MLP':
            data, target = batch['data'].to(self.device), batch['target'].to(self.device)
            patient_id = batch['id']
            outputs = model(data)

        elif self.model_name == 'Hypergraph':
            batch.to(self.device)
            target = batch.y
            patient_id = batch.patient_id
            outputs = model(batch)

        return outputs, target, patient_id

    def analyze_biomarkers(self, model, test_loader):
        """
        Perform biomarker analysis using the trained model.

        Args:
            model: Trained model
            test_loader: Test data loader

        Returns:
            Dictionary of biomarker analysis results
        """
        # Only applicable for hypergraph model with pathway information
        if self.model_name != 'Hypergraph' or not hasattr(model, 'get_pathway_importance'):
            self.logger.logger.warning("Biomarker analysis only available for Hypergraph model")
            return None

        from legacy_code.utils.biomarker_analysis import BiomarkerAnalysis

        # Initialize biomarker analyzer
        analyzer = BiomarkerAnalysis(
            model=model,
            gene_names=self.hypergraph_data['gene_names'],
            pathway_names=self.hypergraph_data['pathway_names'],
            label_dict=self.label_dict
        )

        # Collect patient-level importance scores
        analyzer.collect_patient_importance(test_loader, self.device)

        # Analyze by patient groups
        group_results = analyzer.analyze_by_group()

        # Identify differential biomarkers
        biomarker_results = analyzer.identify_differential_biomarkers()

        # Generate report
        biomarker_dir = os.path.join(self.test_results_dir, 'biomarker_analysis')
        report_paths = analyzer.generate_complete_report(biomarker_dir)

        return {
            'analyzer': analyzer,
            'group_results': group_results,
            'biomarker_results': biomarker_results,
            'report_paths': report_paths
        }

    def visualize_results(self, metrics):
        """
        Generate visualizations of gene expression test results.

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

        # 2. ROC curve
        if metrics['all_probs'].shape[1] == 2:  # Binary classification
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

    def run_testing(self, test_data, aux_test_data=None, checkpoint_path=None):
        """
        Run complete testing process for gene expression models.

        Args:
            test_data: Gene expression test data
            checkpoint_path: Path to model checkpoint (optional)

        Returns:
            Dictionary of test results and paths to outputs
        """
        # Call parent method for standard evaluation

        results = super().run_testing(test_data, checkpoint_path=checkpoint_path)

        if self.config['execution'].get('visualise', False):
            # Add biomarker analysis if applicable
            if self.model_name == 'Hypergraph' and hasattr(results['model'], 'get_pathway_importance'):
                self.logger.logger.info("Performing biomarker analysis")
                biomarker_results = self.analyze_biomarkers(results['model'],
                                                            self.prepare_test_data(test_data))

            if biomarker_results:
                results['biomarker_analysis'] = biomarker_results
        else:
            self.logger.logger.info("Skipping biomarker analysis and visualizations")

        return results