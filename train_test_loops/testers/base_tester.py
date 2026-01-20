# train_test_loops/testers/base_tester.py

import os
import pandas as pd

import torch

from abc import ABC, abstractmethod

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sksurv.metrics import concordance_index_censored
from utils.survival_utils import calculate_risk


class BaseTester(ABC):
    """
    Abstract base class for all testers in the ProtoPathway framework.
    Handles model evaluation and result visualization.
    """

    def __init__(
            self,
            config,
            experiment_logger,
            is_continuation=False,
            device=None
    ):
        """
        Initialize the base tester.

        Args:
            config: Dictionary containing configuration parameters
            experiment_logger: Logger instance for the experiment
            device: Computation device (CPU/GPU)
        """
        self.config = config
        self.logger = experiment_logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if is_continuation:
            # Use the paths from the experiment_logger
            self.results_dir = experiment_logger.log_dir
            self.checkpoint_dir = experiment_logger.checkpoint_dir
        else:
            # Use the paths from the testing configuration
            self.results_dir = config.get('testing', {}).get('experiment_path', '')
            if not os.path.exists(self.results_dir):
                raise FileNotFoundError(f"Experiment directory not found: {self.results_dir}")
            self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')

        # Create test results directory
        self.test_results_dir = os.path.join(self.results_dir, 'test_results')
        os.makedirs(self.test_results_dir, exist_ok=True)

        # Label mapping
        self.label_dict = config.get('label_dict', {})

        # Check if the task is survival
        self.is_survival = config.get('execution', {}).get('task', 'classification') == 'survival'
        self.is_visualise = config['execution']['visualise']

    @abstractmethod
    def prepare_test_data(self, test_data, aux_test_data=None):
        """
        Prepare test data loader.

        Args:
            test_data: Test data

        Returns:
            test_loader
        """
        pass

    @abstractmethod
    def load_model(self, checkpoint_path):
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Loaded model
        """
        pass


    def evaluate(self, model, test_loader, aux_test_loader=None):
        """
        Evaluate model on test data.

        Args:
            model: Trained model
            test_loader: Test data loader

        Returns:
            Dictionary of test metrics
        """
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        all_patient_ids = []
        attention_dict = {}

        # Survival-specific collections
        if self.is_survival:
            all_risk_scores = []
            all_survival_times = []
            all_censorships = []
        else:
            # Classification-specific
            correct = 0
            total = 0

        # Determine if this is multimodal
        is_multimodal = aux_test_loader is not None

        with torch.no_grad():
            for batch in test_loader:
                if is_multimodal:
                    # Process batch with auxiliary data
                    result = self._process_batch(model, batch, aux_test_loader)
                else:
                    # Process batch without auxiliary data
                    result = self._process_batch(model, batch)

                if self.is_survival:
                    outputs, targets, patient_id, survival_times, censorships, attention_results = result
                    attention_dict[patient_id] = attention_results
                    # Calculate risk scores
                    risk_scores, _ = calculate_risk(outputs)
                    all_risk_scores.append(risk_scores)
                    all_survival_times.append(survival_times)
                    all_censorships.append(censorships)
                else:
                    outputs, targets, patient_id, attention_results = result

                    attention_dict[patient_id[0]] = attention_results

                    # Calculate probabilities and predictions
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)

                    # Store results
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())
                    all_probs.append(probs.cpu())

                    # Calculate accuracy
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

                all_patient_ids.extend(patient_id)

        if self.is_visualise:
            attention_dict['pathway_idx'] = batch['pathway_idx']
            attention_dict['gene_idx'] = batch['gene_idx']

        # Calculate final metrics
        if self.is_survival:
            # Concatenate survival results
            all_risk_scores = torch.cat(all_risk_scores, dim=0).cpu().detach().numpy()
            all_survival_times = torch.cat(all_survival_times, dim=0).cpu().detach().numpy()
            all_censorships = torch.cat(all_censorships, dim=0).cpu().detach().numpy()

            # Calculate c-index (convert censorship: 0=censored becomes True=event_occurred)
            event_occurred = ~all_censorships.astype(bool)
            c_index = concordance_index_censored(event_occurred, all_survival_times, all_risk_scores)

            metrics = {
                'c_index': c_index[0],
                'all_risk_scores': all_risk_scores,
                'all_survival_times': all_survival_times,
                'all_censorships': all_censorships,
                'patient_ids': all_patient_ids,
                'attention_dict': attention_dict
            }
        else:
            # Concatenate classification results
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            all_probs = torch.cat(all_probs, dim=0).numpy()

            # Calculate metrics
            metrics = self._calculate_metrics(all_targets, all_preds, all_probs)
            metrics['patient_ids'] = all_patient_ids
            metrics['all_preds'] = all_preds
            metrics['all_targets'] = all_targets
            metrics['all_probs'] = all_probs
            metrics['attention_dict'] = attention_dict

        return metrics

    @abstractmethod
    def _process_batch(self, model, batch, wsi_data=None):
        """
        Process a batch of data for evaluation.

        Args:
            model: The model being evaluated
            batch: A batch of data

        Returns:
            Tuple of (outputs, targets, patient_ids)
        """
        pass

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        """
        Calculate evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        n_classes = self.config['num_classes']

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,  # as percentage
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred,
                                                           output_dict=True, zero_division=0)
        }

        # Calculate AUC for binary and multiclass
        if n_classes == 2:
            metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_probs,
                                               average='macro', multi_class='ovr')
            except ValueError:
                metrics['auc'] = 0.0
                self.logger.logger.warning("Could not calculate AUC for multiclass problem")

        return metrics

    def save_predictions(self, metrics, output_path=None):
        """
        Save patient-level predictions to CSV.

        Args:
            metrics: Dictionary containing evaluation metrics and predictions
            output_path: Path to save predictions (default: test_results_dir/predictions.csv)

        Returns:
            Path to saved predictions file
        """
        output_path = output_path or os.path.join(self.test_results_dir, 'predictions.csv')

        # Create prediction dataframe
        predictions_df = pd.DataFrame({
            'patient_id': metrics['patient_ids']
        })

        if self.is_survival:
            # Add survival-specific columns
            predictions_df['risk_score'] = metrics['all_risk_scores']
            predictions_df['survival_time'] = metrics['all_survival_times']
            predictions_df['censorship'] = metrics['all_censorships']
        else:
            # Add classification-specific columns
            predictions_df['true_label'] = metrics['all_targets']
            predictions_df['predicted_label'] = metrics['all_preds']

            # Add class names if available
            if self.label_dict:
                predictions_df['true_class'] = predictions_df['true_label'].apply(
                    lambda x: self.label_dict.get(str(x), f"Class_{x}")
                )
                predictions_df['predicted_class'] = predictions_df['predicted_label'].apply(
                    lambda x: self.label_dict.get(str(x), f"Class_{x}")
                )

            # Add probability columns
            for i in range(metrics['all_probs'].shape[1]):
                class_name = self.label_dict.get(str(i), f"Class_{i}")
                predictions_df[f'prob_{class_name}'] = metrics['all_probs'][:, i]

        # Save to CSV
        predictions_df.to_csv(output_path, index=False)
        self.logger.logger.info(f"Saved predictions to {output_path}")

        return output_path



    def save_metrics_report(self, metrics, output_path=None):
        """
        Save evaluation metrics to text file.

        Args:
            metrics: Dictionary containing evaluation metrics
            output_path: Path to save report (default: test_results_dir/metrics_report.txt)

        Returns:
            Path to saved report
        """
        output_path = output_path or os.path.join(self.test_results_dir, 'metrics_report.txt')

        # Format metrics for report
        with open(output_path, 'w') as f:
            f.write("=== ProtoPathway Test Results ===\n\n")
            f.write(f"Dataset: {self.config['dataset_name']}\n")
            f.write(f"Task: {self.config['execution']['task']}\n")
            f.write(f"Mode: {self.config['execution']['mode']}\n\n")

            if self.is_survival:
                f.write("=== Survival Analysis Metrics ===\n")
                f.write(f"C-index: {metrics['c_index']:.4f}\n")
                f.write(f"Number of patients: {len(metrics['patient_ids'])}\n")

                if 'all_censorships' in metrics:
                    n_events = sum(~metrics['all_censorships'].astype(bool))
                    n_censored = sum(metrics['all_censorships'].astype(bool))
                    f.write(f"Events: {n_events}\n")
                    f.write(f"Censored: {n_censored}\n")
                    f.write(f"Event rate: {n_events / (n_events + n_censored) * 100:.1f}%\n")
            else:
                f.write("=== Classification Metrics ===\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"AUC: {metrics['auc']:.4f}\n\n")

                f.write("=== Confusion Matrix ===\n")
                f.write(f"{metrics['confusion_matrix']}\n\n")

                f.write("=== Classification Report ===\n")
                class_report = metrics['classification_report']
                for class_name, values in class_report.items():
                    if isinstance(values, dict):
                        f.write(f"Class {class_name}:\n")
                        for metric_name, value in values.items():
                            f.write(f"  {metric_name}: {value:.4f}\n")
                        f.write("\n")

        self.logger.logger.info(f"Saved metrics report to {output_path}")

        return output_path


    def visualize_results(self, metrics):
        """
        Generate visualizations of test results.

        Args:
            metrics: Dictionary containing evaluation metrics

        Returns:
            Dictionary of paths to visualization files
        """
        # This will be implemented by each modality-specific tester
        # as visualization needs differ by modality
        pass


    def run_testing(self, test_data, aux_test_data=None, checkpoint_path=None):
        """
        Run complete testing process.

        Args:
            test_data: Test data
            checkpoint_path: Path to model checkpoint (optional)

        Returns:
            Dictionary of test results and paths to outputs
        """
        # Prepare test data
        if aux_test_data is not None:
            # Multimodal case
            test_loaders = self.prepare_test_data(test_data, aux_test_data)
            if not isinstance(test_loaders, tuple):
                raise ValueError("Multimodal prepare_test_data should return a tuple of loaders")
        else:
            # Single-modality case
            test_loader = self.prepare_test_data(test_data)
            test_loaders = (test_loader,)

        # Find best checkpoint if not specified
        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        self.logger.logger.info(f"Loading model from {checkpoint_path}")

        # Load model
        model, criterion = self.load_model(checkpoint_path)

        if aux_test_data is not None:
            ge_test_loader, wsi_test_loader = test_loaders
            metrics = self.evaluate(model, ge_test_loader, wsi_test_loader)
        else:
            metrics = self.evaluate(model, test_loaders)

        # Save predictions and metrics
        predictions_path = self.save_predictions(metrics)
        report_path = self.save_metrics_report(metrics)

        # Generate visualizations
        visualization_paths = self.visualize_results(metrics)

        # Log summary
        if self.is_survival:
            self.logger.logger.info(f"Test C-index: {metrics.get('c_index', 0):.4f}")
        else:
            self.logger.logger.info(f"Test Accuracy: {metrics.get('accuracy', 0):.2f}%")
            self.logger.logger.info(f"Test F1 Score: {metrics.get('f1', 0):.4f}")
            self.logger.logger.info(f"Test AUC: {metrics.get('auc', 0):.4f}")

        # Return results
        return {
            'metrics': metrics,
            'output_paths': {
                'predictions': predictions_path,
                'report': report_path,
                'visualizations': visualization_paths
            },
            'model': model
        }