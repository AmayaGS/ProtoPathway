
import os

import torch
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the ProtoPathway framework.
    """

    def __init__(
            self,
            config,
            experiment_logger,
            device=None
    ):
        """
        Initialize the base trainer.

        Args:
            config: Dictionary containing configuration parameters
            experiment_logger: Logger instance for the experiment
            device: Computation device (CPU/GPU)
        """
        self.config = config
        self.logger = experiment_logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Common parameters
        self.num_epochs = config['ge_training']['num_epochs']
        self.checkpoint = config['ge_training']['checkpoint']
        self.weight_type = config['ge_training']['weight_type']
        self.metric_to_track = 'loss' if self.weight_type == 'loss' else 'acc' if self.weight_type == 'accuracy' else 'auc'
        self.mode = 'min' if self.weight_type == 'loss' else 'max'

        # Paths for saving results
        self.results_dir = experiment_logger.log_dir
        self.checkpoint_dir = experiment_logger.checkpoint_dir

        # Metrics tracking
        self.best_metrics = {
            'acc': 0.0,
            'auc': 0.0,
            'precision': 0.0,
            'loss': float('inf'),
            'epoch': 0
        }

        # Thresholds for significant improvement
        self.thresholds = {
            'acc': 0.,
            'auc': 0.,
            'precision': 0.,
            'loss': 0.05
        }

    @abstractmethod
    def prepare_data(self, train_data, val_data):
        """
        Prepare data loaders for ge_training and validation.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            train_loader, val_loader
        """
        pass

    @abstractmethod
    def create_model(self):
        """
        Create and initialize the model.

        Returns:
            model, criterion, optimizer, lr_scheduler
        """
        pass

    @abstractmethod
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """
        Run one ge_training epoch.

        Args:
            model: The model to train
            train_loader: DataLoader for ge_training data
            optimizer: The optimizer
            criterion: Loss function

        Returns:
            Dictionary of metrics for this epoch
        """
        pass

    @abstractmethod
    def validate(self, model, val_loader, criterion):
        """
        Validate the model.

        Args:
            model: The model to validate
            val_loader: DataLoader for validation data
            criterion: Loss function

        Returns:
            Dictionary of validation metrics
        """
        pass

    def should_save_model(self, metrics):
        """
        Determines if the model should be saved based on performance metrics.

        Args:
            metrics: Dictionary of validation metrics

        Returns:
            Tuple of (should_save, reason)
        """
        # Extract current metrics
        curr_acc = metrics['acc']
        curr_auc = metrics.get('auc', 0.0)
        curr_precision = metrics.get('precision', 0.0)
        curr_loss = metrics['loss']

        # Get best metrics so far
        best_acc = self.best_metrics['acc']
        best_auc = self.best_metrics['auc']
        best_precision = self.best_metrics['precision']
        best_loss = self.best_metrics['loss']

        # Get thresholds
        acc_threshold = self.thresholds['acc']
        auc_threshold = self.thresholds['auc']
        precision_threshold = self.thresholds['precision']
        loss_threshold = self.thresholds['loss']

        # Calculate if metrics are significantly better or approximately equal
        acc_significantly_better = curr_acc > best_acc + acc_threshold
        acc_approximately_same = abs(curr_acc - best_acc) <= acc_threshold

        auc_better = curr_auc > best_auc
        auc_approximately_same = abs(curr_auc - best_auc) <= auc_threshold

        precision_better = curr_precision > best_precision
        precision_approximately_same = abs(curr_precision - best_precision) <= precision_threshold

        # For loss, lower is better, and we check for relative improvement
        loss_significantly_better = curr_loss < best_loss * (1 - loss_threshold)

        # Reason for saving (for logging)
        save_reason = None

        # Decision tree
        if acc_significantly_better:
            save_reason = f"Accuracy significantly improved: {curr_acc:.2f}% vs {best_acc:.2f}%"
            return True, save_reason

        elif acc_approximately_same:
            if auc_better:
                save_reason = f"Equal accuracy with better AUC: {curr_auc:.4f} vs {best_auc:.4f}"
                return True, save_reason

            elif auc_approximately_same:
                if precision_better:
                    save_reason = f"Equal accuracy & AUC with better precision: {curr_precision:.4f} vs {best_precision:.4f}"
                    return True, save_reason

                elif precision_approximately_same and loss_significantly_better:
                    save_reason = f"Equal accuracy, AUC & precision with significantly better loss: {curr_loss:.4f} vs {best_loss:.4f}"
                    return True, save_reason

        # Default case - no saving
        return False, save_reason

    def train(self, train_data, val_data, fold_idx=None):
        """
        Main ge_training loop.

        Args:
            train_data: Training data
            val_data: Validation data
            fold_idx: Index of the current fold (for cross-validation)

        Returns:
            Tuple of (trained_model, training_history)
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(train_data, val_data)

        # Create model and optimization components
        model, criterion, optimizer, lr_scheduler = self.create_model()

        # Setup checkpoint name
        self.checkpoint_name = f"best_fold_{fold_idx}.pt" if fold_idx is not None else "best_model.pt"

        # Initialize history
        history = {
            'train': {
                'loss': [],
                'acc': []
            },
            'val': {
                # Initialize complex metrics as empty dictionaries
                'confusion_matrix': {},
                'all_labels': {},
                'all_probs': {},
                'classification_report': {}
            },
            'best_epoch': None,
            'best_metrics': None
        }

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            self.logger.start_timer(f"epoch_{epoch}")

            # Train for one epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            self.logger.log_metrics(train_metrics, epoch, 'train')

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)
            self.logger.log_metrics(val_metrics, epoch, 'val')

            # Update ge_training history
            for metric, value in train_metrics.items():
                if metric not in history['train']:
                    history['train'][metric] = []
                history['train'][metric].append(value)

            for metric, value in val_metrics.items():
                # Store complex metrics by epoch
                if metric in ['confusion_matrix', 'all_labels', 'all_probs', 'classification_report']:
                    history['val'][metric][epoch] = value
                # Store scalar metrics as lists across epochs
                else:
                    if metric not in history['val']:
                        history['val'][metric] = []
                    history['val'][metric].append(value)

            # Update learning rate scheduler if configured
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_metrics[self.metric_to_track] * (-1 if self.mode == 'max' else 1))
                else:
                    lr_scheduler.step()

            # Check if we should save this model
            should_save, save_reason = self.should_save_model(val_metrics)

            if should_save:
                # Update best metrics
                self.best_metrics['acc'] = val_metrics['acc']
                self.best_metrics['auc'] = val_metrics.get('auc', 0.0)
                self.best_metrics['precision'] = val_metrics.get('precision', 0.0)
                self.best_metrics['loss'] = val_metrics['loss']
                self.best_metrics['epoch'] = epoch

                # Update history
                history['best_epoch'] = epoch
                history['best_metrics'] = self.best_metrics.copy()

                if self.checkpoint:
                    path = os.path.join(self.logger.checkpoint_dir, self.checkpoint_name)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metrics': self.best_metrics,
                    }, path)
                    print(f"Saved best model to {path}")
                    print(f"Reason: {save_reason}")

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

            print(f"Best val {self.metric_to_track}: {self.best_metrics[self.metric_to_track]:.4f}\n" + "-" * 50)

        return model, history