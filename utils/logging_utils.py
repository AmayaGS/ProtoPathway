"""
ProtoPathway: Experiment logging and tracking utilities.
"""

import os
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from typing import Dict, List, Union, Optional, Any, Tuple


class ExperimentLogger:
    """
    Centralized logging system for ML experiments with PyTorch.
    Handles metrics tracking, model checkpoints, hyperparameters, and visualizations.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            experiment_name: Optional[str] = None,
            log_dir: Optional[str] = None,
            save_config: bool = True,
            use_tensorboard: bool = False,
            use_wandb: bool = False,
            use_comet: bool = False,
            comet_project: Optional[str] = None,
            comet_workspace: Optional[str] = None,
            wandb_project: Optional[str] = None
    ):
        """
        Initialize the experiment logger.

        Args:
            config: Configuration dictionary.
            experiment_name: Name for this experiment run.
            log_dir: Directory for saving logs.
            save_config: Whether to save the config as YAML.
            use_tensorboard: Whether to use TensorBoard for logging.
            use_wandb: Whether to use Weights & Biases for logging.
            use_comet: Whether to use Comet.ml for logging.
            comet_project: Comet.ml project name.
            comet_workspace: Comet.ml workspace name.
            wandb_project: W&B project name.
        """
        self.config = config
        self.experiment_name = experiment_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set up directory structure
        self.log_dir = log_dir or os.path.join(config['output_dir'], 'logs', self.experiment_name)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        self.plots_dir = os.path.join(self.log_dir, 'plots')
        self.results_dir = os.path.join(self.log_dir, 'results')

        # Create directories
        for directory in [self.log_dir, self.checkpoint_dir, self.plots_dir, self.results_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Initialize metrics tracking
        self.metrics = {
            'train': {},
            'val': {},
            'test': {}
        }

        # Log hyperparameters
        self.hyperparams = self._extract_hyperparams(config)

        # Save config if requested
        if save_config:
            self._save_config()

        # Set up TensorBoard if requested
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(self.tensorboard_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            except ImportError:
                print("TensorBoard not installed. Run: pip install tensorboard")
                self.use_tensorboard = False

        # Set up Weights & Biases if requested
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "protopathway",
                    name=self.experiment_name,
                    config=self.hyperparams
                )
                self.wandb = wandb
            except ImportError:
                print("Weights & Biases not installed. Run: pip install wandb")
                self.use_wandb = False

        # Set up Comet.ml if requested
        self.use_comet = use_comet
        if use_comet:
            try:
                from comet_ml import Experiment

                experiment = Experiment(
                    project_name=comet_project or "protopathway",
                    workspace=comet_workspace,
                    auto_metric_logging=False,
                    auto_param_logging=False
                )

                # Set experiment name
                experiment.set_name(self.experiment_name)

                # Log hyperparameters
                experiment.log_parameters(self.hyperparams)

                # Store experiment instance
                self.comet_experiment = experiment
                print(f"Comet.ml experiment initialized: {experiment.get_key()}")

            except ImportError:
                print("Comet.ml not installed. Run: pip install comet_ml")
                self.use_comet = False

        # Initialize timers
        self.timers = {}

        print(f"ExperimentLogger initialized at {self.log_dir}")

    def _extract_hyperparams(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hyperparameters from config for logging."""
        hyperparams = {}

        # Extract relevant sections from config
        sections = ['dataset', 'GNN', 'training', 'model', 'graph', 'parameters']
        for section in sections:
            if section in config:
                hyperparams.update({f"{section}.{k}": v for k, v in config[section].items()})

        return hyperparams

    def _save_config(self) -> None:
        """Save the configuration to disk."""
        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name not in self.timers:
            print(f"Timer '{name}' was not started.")
            return 0.0

        elapsed = time.time() - self.timers[name]
        if self.use_tensorboard:
            self.writer.add_scalar(f'time/{name}', elapsed)
        if self.use_wandb:
            self.wandb.log({f'time/{name}': elapsed})

        return elapsed

    def log_metric(self,
                   name: str,
                   value: Union[float, torch.Tensor],
                   step: int,
                   phase: str = 'train') -> None:
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Step/epoch number
            phase: 'train', 'val', or 'test'
        """
        # Convert torch tensors to numpy/python types
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().numpy()

        # Initialize metric list if first time seeing this metric
        if name not in self.metrics[phase]:
            self.metrics[phase][name] = []

        # Append value
        self.metrics[phase][name].append((step, value))

        # Log to TensorBoard if enabled
        if self.use_tensorboard:
            self.writer.add_scalar(f'{phase}/{name}', value, step)

        # Log to Weights & Biases if enabled
        if self.use_wandb:
            self.wandb.log({f'{phase}/{name}': value}, step=step)

        # Log to Comet.ml if enabled
        if self.use_comet:
            metric_name = f'{phase}/{name}'
            self.comet_experiment.log_metric(metric_name, value, step=step)

    def log_metrics(self,
                    metrics_dict: Dict[str, float],
                    step: int,
                    phase: str = 'train') -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric_name: value
            step: Step/epoch number
            phase: 'train', 'val', or 'test'
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, phase)

    def save_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        metrics: Dict[str, float],
                        filename: Optional[str] = None) -> str:
        """
        Save a model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics to save with checkpoint
            filename: Optional custom filename

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        path = os.path.join(self.checkpoint_dir, filename)

        # Prepare state dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        # Save checkpoint
        torch.save(checkpoint, path)

        # Log to Weights & Biases if enabled
        if self.use_wandb:
            if epoch % 10 == 0 or epoch == 1:  # Save model artifacts only occasionally
                self.wandb.save(path)

        return path

    def load_checkpoint(self,
                        model: torch.nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        path: Optional[str] = None,
                        device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load a model checkpoint.

        Args:
            model: PyTorch model to load weights into
            optimizer: Optional PyTorch optimizer to load state into
            path: Path to checkpoint, if None loads latest checkpoint
            device: Device to load the model to

        Returns:
            Tuple of (model, checkpoint_data)
        """
        if path is None:
            # Find latest checkpoint
            checkpoints = list(Path(self.checkpoint_dir).glob('*.pt'))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

            # Sort by modification time (newest first)
            checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
            path = str(checkpoints[0])

        device = device or next(model.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, checkpoint

    def save_best_model(self,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        metrics: Dict[str, float],
                        metric_name: str,
                        mode: str = 'max') -> None:
        """
        Save the model if it's the best so far according to the specified metric.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            metric_name: Name of metric to compare
            mode: 'max' or 'min' - whether higher or lower is better
        """
        if metric_name not in metrics:
            print(f"Warning: Metric {metric_name} not found in metrics. Cannot save best model.")
            return

        current_value = metrics[metric_name]
        best_filename = f"best_{metric_name}.pt"
        best_path = os.path.join(self.checkpoint_dir, best_filename)

        # Check if this is the best model so far
        is_best = False

        if not os.path.exists(best_path):
            is_best = True
        else:
            best_checkpoint = torch.load(best_path, map_location='cpu')
            if mode == 'max':
                is_best = current_value > best_checkpoint['metrics'][metric_name]
            else:  # mode == 'min'
                is_best = current_value < best_checkpoint['metrics'][metric_name]

        if is_best:
            print(f"\n==> New best model with {metric_name} = {current_value:.4f}")
            self.save_checkpoint(model, optimizer, epoch, metrics, best_filename)

    def log_figure(self,
                   figure: plt.Figure,
                   name: str,
                   step: Optional[int] = None) -> str:
        """
        Save a matplotlib figure and optionally log to TensorBoard/W&B/Comet.

        Args:
            figure: Matplotlib figure
            name: Figure name
            step: Optional step/epoch number for TensorBoard

        Returns:
            Path to saved figure
        """
        # Ensure proper file extension
        if not name.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            name = f"{name}.png"

        # Save the figure
        path = os.path.join(self.plots_dir, name)
        figure.savefig(path, bbox_inches='tight')
        plt.close(figure)

        # Log to TensorBoard if enabled and step is provided
        if self.use_tensorboard and step is not None:
            self.writer.add_figure(name, figure, step)

        # Log to Weights & Biases if enabled
        if self.use_wandb:
            self.wandb.log({name: self.wandb.Image(path)}, step=step)

        # Log to Comet.ml if enabled
        if self.use_comet:
            if step is not None:
                self.comet_experiment.log_image(path, name=name, step=step)
            else:
                self.comet_experiment.log_image(path, name=name)

        return path

    def plot_metric(self,
                    metric_name: str,
                    phases: List[str] = None,
                    window_size: int = 1,
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot a metric's history.

        Args:
            metric_name: Name of the metric to plot
            phases: List of phases to include ('train', 'val', 'test')
            window_size: Moving average window size
            figsize: Figure dimensions

        Returns:
            Matplotlib figure
        """
        phases = phases or ['train', 'val']
        fig, ax = plt.subplots(figsize=figsize)

        for phase in phases:
            if metric_name not in self.metrics[phase]:
                continue

            data = self.metrics[phase][metric_name]
            steps, values = zip(*data)

            # Apply moving average if requested
            if window_size > 1:
                values = pd.Series(values).rolling(window=window_size).mean().tolist()

            ax.plot(steps, values, label=f"{phase}")

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs. Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def save_metrics_to_csv(self) -> Dict[str, str]:
        """
        Save all tracked metrics to CSV files.

        Returns:
            Dictionary mapping phase to saved CSV path
        """
        csv_paths = {}

        for phase in self.metrics:
            if not self.metrics[phase]:
                continue

            # Convert metrics to DataFrame
            data = {}
            steps = set()

            for metric_name, values in self.metrics[phase].items():
                steps.update([step for step, _ in values])
                data[metric_name] = {step: value for step, value in values}

            steps = sorted(steps)
            df = pd.DataFrame(index=steps)

            for metric_name, step_values in data.items():
                df[metric_name] = df.index.map(lambda x: step_values.get(x, np.nan))

            # Save to CSV
            csv_path = os.path.join(self.results_dir, f"{phase}_metrics.csv")
            df.to_csv(csv_path)
            csv_paths[phase] = csv_path

        return csv_paths

    def log_hyperparam_comparison(self,
                                  best_metric_name: str,
                                  best_metric_mode: str = 'max',
                                  top_k: int = 5) -> pd.DataFrame:
        """
        Compare this run with previous runs based on hyperparameters.
        Loads previous runs from disk and shows top_k runs by the specified metric.

        Args:
            best_metric_name: Metric to sort by
            best_metric_mode: 'max' or 'min'
            top_k: Number of top runs to show

        Returns:
            DataFrame with the comparison
        """
        log_root = Path(self.log_dir).parent

        # Prepare data for current run
        current_run = {
            'run_name': self.experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **self.hyperparams
        }

        # Add metrics for current run
        for phase in ['train', 'val', 'test']:
            for metric_name in self.metrics[phase]:
                if self.metrics[phase][metric_name]:
                    steps, values = zip(*self.metrics[phase][metric_name])
                    current_run[f'{phase}_{metric_name}_last'] = values[-1]
                    current_run[f'{phase}_{metric_name}_best'] = max(values) if best_metric_mode == 'max' else min(
                        values)

        # Load previous runs
        runs = [current_run]

        for run_dir in log_root.iterdir():
            if not run_dir.is_dir() or run_dir.name == self.experiment_name:
                continue

            config_path = run_dir / 'config.yaml'
            if not config_path.exists():
                continue

            try:
                # Load config
                with open(config_path, 'r') as f:
                    run_config = yaml.safe_load(f)

                # Extract hyperparameters
                run_data = {
                    'run_name': run_dir.name,
                    'timestamp': datetime.fromtimestamp(os.path.getctime(config_path)).strftime('%Y-%m-%d %H:%M:%S'),
                    **self._extract_hyperparams(run_config)
                }

                # Load metrics if available
                results_dir = run_dir / 'results'
                for phase in ['train', 'val', 'test']:
                    metrics_path = results_dir / f"{phase}_metrics.csv"
                    if metrics_path.exists():
                        metrics_df = pd.read_csv(metrics_path)
                        for metric_name in metrics_df.columns[1:]:  # Skip index column
                            run_data[f'{phase}_{metric_name}_last'] = metrics_df[metric_name].iloc[-1]
                            run_data[f'{phase}_{metric_name}_best'] = metrics_df[
                                metric_name].max() if best_metric_mode == 'max' else metrics_df[metric_name].min()

                runs.append(run_data)
            except Exception as e:
                print(f"Error loading run {run_dir.name}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(runs)

        # Sort by specified metric
        metric_col = f'val_{best_metric_name}_best'
        if metric_col in df.columns:
            df = df.sort_values(by=metric_col, ascending=(best_metric_mode == 'min'))

        # Select top_k runs
        if len(df) > top_k:
            df = df.head(top_k)

        # Save comparison to disk
        comparison_path = os.path.join(self.results_dir, 'run_comparison.csv')
        df.to_csv(comparison_path, index=False)

        return df

    def finalize(self) -> None:
        """
        Finalize the experiment logging.
        - Save all metrics to CSV
        - Create final plots
        - Close TensorBoard writer
        - Finalize W&B run
        - End Comet.ml experiment
        """
        # Save metrics to CSV
        self.save_metrics_to_csv()

        # Create plots for all metrics
        for phase in self.metrics:
            for metric_name in self.metrics[phase]:
                if self.metrics[phase][metric_name]:
                    try:
                        fig = self.plot_metric(metric_name)
                        self.log_figure(fig, f"{metric_name}_history.png")
                    except Exception as e:
                        print(f"Error plotting {metric_name}: {e}")

        # Close TensorBoard writer if enabled
        if self.use_tensorboard and hasattr(self, 'writer'):
            self.writer.close()

        # Finalize W&B run if enabled
        if self.use_wandb:
            self.wandb.finish()

        # End Comet.ml experiment if enabled
        if self.use_comet and hasattr(self, 'comet_experiment'):
            # Log any additional information before ending
            self.comet_experiment.log_asset(os.path.join(self.log_dir, 'config.yaml'))

            # Upload CSV files with metrics
            for phase in ['train', 'val', 'test']:
                csv_path = os.path.join(self.results_dir, f"{phase}_metrics.csv")
                if os.path.exists(csv_path):
                    self.comet_experiment.log_asset(csv_path)

            # End the experiment
            self.comet_experiment.end()

        print(f"Experiment completed. Results saved to {self.log_dir}")