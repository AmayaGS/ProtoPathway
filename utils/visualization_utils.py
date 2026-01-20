"""
ProtoPathway: Comprehensive visualization utilities for model evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

from sklearn.preprocessing import label_binarize
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union

from utils.helpers import ensure_directory
from utils.survival_utils import stratify_risk_groups, prepare_km_data

# ============================== INDIVIDUAL PLOT FUNCTIONS ==============================

def plot_kaplan_meier_curves(
        survival_times,
        censorships,
        risk_scores,
        output_path,
        num_groups=2,
        figsize=(10, 8)
):
    """
    Plot Kaplan-Meier survival curves for different risk groups.

    Args:
        survival_times: numpy array of survival times
        censorships: numpy array of censorship status (0=censored, 1=event)
        risk_scores: numpy array of risk scores or predicted risks
        output_path: path to save the plot
        num_groups: number of risk groups (2 or 4)
        figsize: figure size (width, height)

    Returns:
        Dictionary with log-rank test p-value and mean survival times
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test
        from matplotlib.lines import Line2D
    except ImportError:
        print("lifelines package is required for survival analysis. Please install it.")
        return {}

    # Get risk group assignments
    risk_groups = stratify_risk_groups(risk_scores, num_groups=num_groups)

    # Prepare data for KM curves
    km_data = prepare_km_data(survival_times, censorships, risk_groups)

    # Create the plot
    plt.figure(figsize=figsize)

    # Define group labels and colors - low risk is now green
    if num_groups == 2:
        group_names = ['Low Risk', 'High Risk']
        colors = ['green', 'red']
    else:  # num_groups == 4
        group_names = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk']
        colors = ['darkgreen', 'lightgreen', 'orange', 'red']

    # Calculate mean survival times and standard deviations
    mean_survivals = {}

    # Fit and plot KM curves for each risk group
    kmf = KaplanMeierFitter()
    for i, group in enumerate(sorted(km_data.keys())):
        data = km_data[group]
        label = group_names[group]
        kmf.fit(data['durations'], data['event_observed'], label=label)

        # Plot with markers for patients
        ax = kmf.plot(ci_show=True, color=colors[group], show_censors=True,
                      censor_styles={'ms': 6, 'marker': '|'})

        # Calculate mean survival time and std
        # For observed cases (excluding censored data)
        observed_times = data['durations'][data['event_observed']]
        if len(observed_times) > 0:
            mean_survival = np.mean(observed_times)
            std_survival = np.std(observed_times)
        else:
            # If no events were observed, use restricted mean survival time
            survival_curve = kmf.survival_function_
            times = survival_curve.index.values
            probabilities = survival_curve.values.flatten()

            # Calculate RMST (area under the curve)
            rmst = 0
            for j in range(1, len(times)):
                rmst += (times[j] - times[j - 1]) * probabilities[j - 1]

            mean_survival = rmst
            std_survival = None

        mean_survivals[label] = {
            'mean': mean_survival,
            'std': std_survival
        }

    # Calculate log-rank test p-value
    if num_groups == 2:
        # Perform log-rank test for 2 groups
        low_risk = km_data[0]
        high_risk = km_data[1]

        results = logrank_test(
            low_risk['durations'], high_risk['durations'],
            low_risk['event_observed'], high_risk['event_observed']
        )

        p_value = results.p_value
    else:
        # For multiple groups, use multivariate log-rank test
        durations = []
        event_observed = []
        groups = []

        for group, data in km_data.items():
            durations.extend(data['durations'])
            event_observed.extend(data['event_observed'])
            groups.extend([group] * len(data['durations']))

        results = multivariate_logrank_test(
            np.array(durations),
            np.array(event_observed),
            np.array(groups)
        )
        p_value = results.p_value

    # Create custom legend with horizontal lines
    legend_elements = []
    for i, group in enumerate(sorted(km_data.keys())):
        label = group_names[i]
        mean_data = mean_survivals.get(label, {})

        # if mean_data.get('mean') is not None:
        #     if mean_data.get('std') is not None:
        #         label_text = f"{label}: mean {mean_data['mean']:.1f} ± {mean_data['std']:.1f} months"
        #     else:
        #         label_text = f"{label}: mean {mean_data['mean']:.1f} months"
        # else:
        #     label_text = f"{label}: mean N/A"

        # Create a horizontal line for legend
        line = Line2D([0], [0], color=colors[i], lw=2, label=label)
        legend_elements.append(line)

    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    # Add p-value as scientific notation in title with asterisk if significant
    title = f"log-rank p-value = {p_value:.2e}"
    if p_value < 0.05:
        title += "*"  # Add asterisk for significant p-values

    plt.title(title, fontsize=14)

    # Update axis labels
    plt.xlabel('Time [months]', fontsize=12)
    plt.ylabel('Proportion Surviving', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    file_format = 'pdf'
    if not output_path.lower().endswith(f'.{file_format}'):
        output_path = f"{os.path.splitext(output_path)[0]}.{file_format}"

    # Save the plot
    ensure_directory(os.path.dirname(output_path))
    plt.savefig(
        output_path,
        bbox_inches='tight',
        transparent=True,  # Optional: transparent background
        metadata={'Creator': 'ProtoPathway'},  # Optional: add metadata
        dpi=300  # Still helpful for embedded rasters if any
    )
    plt.close()

    return {
        'p_value': p_value,
        'mean_survivals': mean_survivals
    }

def plot_confusion_matrix(
        cm: np.ndarray,
        classes: List[str],
        output_path: str,
        title: str = 'Confusion Matrix',
        normalize: bool = False,
        cmap: plt.cm = plt.cm.Blues,
        figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        classes: List of class names
        output_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap
        figsize: Figure size (width, height)
    """
    # Make a copy to avoid modifying the original
    cm_plot = cm.copy()

    # Check if the matrix contains floating point values
    is_float = np.issubdtype(cm_plot.dtype, np.floating) or normalize

    if normalize:
        cm_plot = cm_plot.astype('float') / cm_plot.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        # If it's already float, use float format, otherwise use integer format
        fmt = '.1f' if is_float else 'd'

    plt.figure(figsize=figsize)
    plt.imshow(cm_plot, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Add labels to each cell
    thresh = cm_plot.max() / 2.
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        # Convert to int first if using 'd' format to avoid the error
        value = cm_plot[i, j]
        if fmt == 'd':
            value = int(value)
            plt.text(j, i, f"{value}",
                     horizontalalignment="center",
                     color="white" if cm_plot[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{value:{fmt}}",
                     horizontalalignment="center",
                     color="white" if cm_plot[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))

    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_roc_curves(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        classes: List[str],
        output_path: str,
        title: str = 'ROC Curves',
        figsize: Tuple[int, int] = (12, 10)
) -> Dict[str, float]:
    """
    Plot ROC curves for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        classes: List of class names
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        Dictionary mapping class names to AUC scores
    """
    plt.figure(figsize=figsize)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(classes)

    # Binarize the labels for one-vs-rest ROC
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # For binary classification, ensure y_bin is in the right shape
    if n_classes == 2:
        y_bin = np.hstack((1 - y_bin, y_bin))

    # Plot each class ROC curve
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))

    for i, color, class_name in zip(range(n_classes), colors, classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    # Plot the random guessing line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))

    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Return dictionary mapping class names to AUC scores
    return {class_name: roc_auc[i] for i, class_name in enumerate(classes)}


def plot_precision_recall_curves(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        classes: List[str],
        output_path: str,
        title: str = 'Precision-Recall Curves',
        figsize: Tuple[int, int] = (12, 10)
) -> Dict[str, float]:
    """
    Plot Precision-Recall curves for multi-class classification.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        classes: List of class names
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        Dictionary mapping class names to Average Precision scores
    """
    plt.figure(figsize=figsize)

    # Compute Precision-Recall and AP for each class
    precision = {}
    recall = {}
    avg_precision = {}
    n_classes = len(classes)

    # Binarize the labels for one-vs-rest curves
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # For binary classification, ensure y_bin is in the right shape
    if n_classes == 2:
        y_bin = np.hstack((1 - y_bin, y_bin))

    # Plot each class PR curve
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))

    for i, color, class_name in zip(range(n_classes), colors, classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_bin[:, i], y_probs[:, i])

        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{class_name} (AP = {avg_precision[i]:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))

    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Return dictionary mapping class names to Average Precision scores
    return {class_name: avg_precision[i] for i, class_name in enumerate(classes)}


def plot_learning_curves(
        history: Dict[str, Any],
        output_path: str,
        metrics: List[str] = None,
        best_epoch: Optional[int] = None,
        title_prefix: str = '',
        figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot learning curves from ge_training history.

    Args:
        history: Dictionary containing ge_training history with 'train' and 'val' keys
        output_path: Base path to save the plots
        metrics: List of metrics to plot (defaults to ['loss', 'acc'])
        best_epoch: Best epoch number to highlight on the plots
        title_prefix: Prefix to add to the plot titles
        figsize: Figure size (width, height)
    """
    if metrics is None:
        if 'c-index' in history['val']:
            metrics = ['loss', 'c-index']
        else:
            metrics = ['loss', 'acc']

    for metric in metrics:
        if metric not in history['train'] or metric not in history['val']:
            continue

        plt.figure(figsize=figsize)

        train_values = history['train'][metric]
        val_values = history['val'][metric]
        epochs = range(1, len(train_values) + 1)

        plt.plot(epochs, train_values, 'b-', label=f'Training {metric}')
        plt.plot(epochs, val_values, 'r-', label=f'Validation {metric}')

        # Highlight best epoch if provided
        if best_epoch is not None and best_epoch <= len(val_values):
            best_value = val_values[best_epoch - 1]
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
                        label=f'Best epoch ({best_epoch})')
            plt.plot(best_epoch, best_value, 'go', markersize=10)
            plt.annotate(f'{best_value:.4f}',
                         xy=(best_epoch, best_value),
                         xytext=(best_epoch + 0.5, best_value),
                         fontsize=10)

        metric_display = 'C-index' if metric == 'c_index' else metric.capitalize()
        title = f'{title_prefix}{metric_display} vs. Epochs'
        plt.title(title, fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(metric_display, fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        if metric == 'c_index':
            plt.ylim([0, 1.05])

        # Ensure x-axis shows integers for epochs
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save plot
        metric_path = os.path.join(output_path, f'{metric}_curve.png')
        ensure_directory(os.path.dirname(metric_path))
        plt.savefig(metric_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_aggregated_learning_curves(
        histories: List[Dict[str, Any]],
        output_path: str,
        metrics: List[str] = None,
        best_epochs: List[int] = None,
        title_prefix: str = 'Mean ',
        figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot aggregated learning curves with mean and standard deviation.

    Args:
        histories: List of history dictionaries
        output_path: Base path to save the plots
        metrics: List of metrics to plot (defaults to ['loss', 'acc'])
        best_epochs: List of best epoch numbers for each history
        title_prefix: Prefix to add to the plot titles
        figsize: Figure size (width, height)
    """
    if metrics is None:
        if 'c-index' in histories[0]['val']:
            metrics = ['loss', 'c-index']
        else:
            metrics = ['loss', 'acc']

    # Find maximum number of epochs across all histories
    max_epochs = max([len(h['train'].get(metrics[0], [])) for h in histories])

    for metric in metrics:
        plt.figure(figsize=figsize)

        # Initialize arrays for train and val metrics
        train_values = np.zeros((len(histories), max_epochs))
        val_values = np.zeros((len(histories), max_epochs))

        # Fill arrays with values from each history
        for i, history in enumerate(histories):
            if metric not in history['train'] or metric not in history['val']:
                continue

            train_data = history['train'][metric]
            val_data = history['val'][metric]

            # Fill in values from this history
            train_values[i, :len(train_data)] = train_data
            val_values[i, :len(val_data)] = val_data

        # Calculate mean and std
        train_mean = np.nanmean(train_values, axis=0)
        train_std = np.nanstd(train_values, axis=0)
        val_mean = np.nanmean(val_values, axis=0)
        val_std = np.nanstd(val_values, axis=0)

        # Create x-axis values
        epochs = range(1, max_epochs + 1)

        # Plot means with std bands
        plt.plot(epochs, train_mean, 'b-', label=f'Mean Training {metric}')
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         color='b', alpha=0.2, label=f'±1 std dev (train)')

        plt.plot(epochs, val_mean, 'r-', label=f'Mean Validation {metric}')
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         color='r', alpha=0.2, label=f'±1 std dev (val)')

        # Plot best epochs if provided
        if best_epochs:
            for i, best_epoch in enumerate(best_epochs):
                if best_epoch is not None and best_epoch <= max_epochs:
                    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.3)

                    # Add marker at the validation value for this fold at its best epoch
                    if i < len(histories) and best_epoch <= len(histories[i]['val'].get(metric, [])):
                        best_value = histories[i]['val'][metric][best_epoch - 1]
                        plt.plot(best_epoch, best_value, 'o', color='g', alpha=0.5, markersize=4)

            # Add a generic label for best epochs
            plt.plot([], [], 'g--', label='Best epochs')

        title = f'{title_prefix}{metric.capitalize()} vs. Epochs'
        plt.title(title, fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Ensure x-axis shows integers for epochs
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save plot
        metric_path = os.path.join(output_path, f'mean_{metric}_curve.png')
        ensure_directory(os.path.dirname(metric_path))
        plt.savefig(metric_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_metric_comparison_by_fold(
        fold_results: pd.DataFrame,
        output_path: str,
        figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot comparison of metrics across folds.

    Args:
        fold_results: DataFrame containing results for each fold
        output_path: Path to save the plots
        figsize: Figure size (width, height)
    """
    # Get metrics to plot (exclude non-numeric columns)
    metrics = [col for col in fold_results.columns
               if col not in ['fold', 'best_epoch']
               and pd.api.types.is_numeric_dtype(fold_results[col])]

    # Create a directory for fold comparison plots
    fold_comparison_dir = os.path.join(output_path, 'fold_comparisons')
    ensure_directory(fold_comparison_dir)

    # Plot bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=figsize)

        # Create bar chart with error bars
        ax = sns.barplot(x='fold', y=metric, data=fold_results, errorbar=None)

        # Add labels on top of bars
        for i, value in enumerate(fold_results[metric]):
            ax.text(i, value * 1.01, f'{value:.4f}', ha='center', fontsize=9)

        # Add mean line
        mean_value = fold_results[metric].mean()
        plt.axhline(y=mean_value, color='r', linestyle='--',
                    label=f'Mean: {mean_value:.4f}')

        # Add std annotation
        std_value = fold_results[metric].std()
        plt.annotate(f'Std: {std_value:.4f}',
                     xy=(0.02, 0.02),
                     xycoords='axes fraction',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.title(f'{metric.capitalize()} by Fold', fontsize=14)
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save plot
        metric_path = os.path.join(fold_comparison_dir, f'{metric}_by_fold.png')
        plt.savefig(metric_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Create a heatmap of all metrics across folds
    plt.figure(figsize=(max(8, len(metrics) * 1.2), max(6, len(fold_results) * 0.8)))

    # Prepare data for heatmap
    heatmap_data = fold_results.copy()
    if 'fold' in heatmap_data.columns:
        heatmap_data['fold'] = heatmap_data['fold'].apply(lambda x: f'Fold {x}')
        heatmap_data = heatmap_data.set_index('fold')

    # Select only numeric columns
    numeric_cols = [col for col in heatmap_data.columns if pd.api.types.is_numeric_dtype(heatmap_data[col])]
    heatmap_data = heatmap_data[numeric_cols]

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=.5)
    plt.title('Metrics by Fold', fontsize=14)
    plt.tight_layout()

    # Save heatmap
    heatmap_path = os.path.join(fold_comparison_dir, 'fold_metrics_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()


# ============================== HELPER FUNCTIONS ==============================

def get_best_epoch_data(
        history: Dict[str, Dict[str, Any]],
        metric_name: str = 'acc',
        mode: str = 'max'
) -> Tuple[int, Dict[str, Any]]:
    """
    Get the best epoch and its data based on a validation metric.

    Args:
        history: Training history dictionary
        metric_name: Metric to use for determining best epoch
        mode: 'max' to maximize metric, 'min' to minimize

    Returns:
        Tuple of (best_epoch, metrics_dict_for_best_epoch)
    """
    # Check if best_epoch is explicitly stored
    if 'best_epoch' in history and history['best_epoch'] is not None:
        best_epoch = history['best_epoch']
    else:
        # Determine best epoch from validation metrics
        val_metrics = history['val'].get(metric_name, [])
        if not val_metrics:
            return None, {}

        if mode == 'max':
            best_idx = np.argmax(val_metrics)
        else:  # mode == 'min'
            best_idx = np.argmin(val_metrics)

        best_epoch = best_idx + 1  # Convert to 1-indexed epoch

    # Get all metrics for this epoch
    best_metrics = {}
    for phase in ['train', 'val']:
        best_metrics[phase] = {}
        for metric, values in history[phase].items():
            if isinstance(values, list) and len(values) >= best_epoch:
                best_metrics[phase][metric] = values[best_epoch - 1]
            elif isinstance(values, dict) and best_epoch in values:
                best_metrics[phase][metric] = values[best_epoch]

    return best_epoch, best_metrics


def create_result_summary(fold_results: pd.DataFrame, config: Dict, output_path: str) -> None:
    """
    Create a comprehensive summary of results.

    Args:
        fold_results: DataFrame containing fold metrics
        config: Configuration dictionary
        output_path: Path to save the summary
    """
    # Calculate mean and std of metrics
    metrics_summary = fold_results.describe().loc[['mean', 'std', 'min', 'max']]

    # Save summary to CSV
    metrics_summary.to_csv(os.path.join(output_path, 'metrics_summary.csv'))

    # Create text summary
    with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
        f.write("ProtoPathway Model Results Summary\n")
        f.write("==================================\n\n")

        f.write(f"Dataset: {config['dataset_name']}\n")
        # f.write(f"Model: {config['model']['name']}\n")

        if 'fold' in fold_results.columns:
            f.write(f"Number of folds: {len(fold_results)}\n\n")

        f.write("Metrics Summary:\n")
        for metric in fold_results.columns:
            if metric not in ['fold', 'best_epoch'] and pd.api.types.is_numeric_dtype(fold_results[metric]):
                mean_val = fold_results[metric].mean()
                std_val = fold_results[metric].std()
                min_val = fold_results[metric].min()
                max_val = fold_results[metric].max()
                f.write(f"  {metric}:\n")
                f.write(f"    Mean ± Std: {mean_val:.4f} ± {std_val:.4f}\n")
                f.write(f"    Min: {min_val:.4f}, Max: {max_val:.4f}\n")

        if 'best_epoch' in fold_results.columns:
            f.write(f"\nAverage best epoch: {fold_results['best_epoch'].mean():.1f}\n")

        # f.write("\nTraining Configuration:\n")
        # f.write(f"  Learning rate: {config['ge_training']['learning_rate']}\n")
        # f.write(f"  Batch size: {config['training']['batch_size']}\n")
        # f.write(f"  Number of epochs: {config['training']['num_epochs']}\n")
        # f.write(f"  Dropout rate: {config['ge_training']['dropout_rate']}\n")
        # f.write(f"  L1 regularization: {config['ge_training']['L1_norm']}\n")
        # f.write(f"  L2 regularization: {config['ge_training']['L2_norm']}\n")
        #
        # f.write("\nModel Configuration:\n")
        # if 'embedding_dim' in config['ge_training']:
        #     f.write(f"  Hidden dimensions: {config['ge_training']['embedding_dim']}\n")
        # if 'num_classes' in config:
        #     f.write(f"  Number of classes: {config['num_classes']}\n")
        # if 'label_dict' in config:
        #     f.write(f"  Class mapping: {config['label_dict']}\n")



# ============================== MAIN VISUALIZATION FUNCTIONS ==============================


def visualize_survival_results(
        y_time,
        y_event,
        risk_scores,
        output_dir,
        title_prefix="",
        num_groups=2
):
    """
    Generate survival-specific visualizations.

    Args:
        y_time: numpy array of survival times
        y_event: numpy array of event indicators
        risk_scores: numpy array of predicted risk scores
        output_dir: directory to save visualizations
        title_prefix: prefix to add to plot titles
        num_groups: number of risk groups for stratification (2 or 4)

    Returns:
        Dictionary with statistics and plot paths
    """
    # Create directory
    survival_dir = os.path.join(output_dir, "survival_analysis")
    ensure_directory(survival_dir)

    # Plot KM curves
    km_path = os.path.join(survival_dir, f"kaplan_meier_{num_groups}_groups.png")
    km_stats = plot_kaplan_meier_curves(
        y_time,
        y_event,
        risk_scores,
        km_path,
        num_groups=num_groups
    )

    # If you want to show both 2-group and 4-group stratifications
    if num_groups != 4:
        km_path_4 = os.path.join(survival_dir, "kaplan_meier_4_groups.png")
        km_stats_4 = plot_kaplan_meier_curves(
            y_time,
            y_event,
            risk_scores,
            km_path_4,
            num_groups=4
        )

    # You could also add other survival-specific visualizations here

    return {
        'km_path': km_path,
        'km_stats': km_stats
    }


def visualize_fold_results(
        fold_data: Dict[str, Any],
        fold_idx: int,
        output_dir: str,
        config: Dict[str, Any],
        metric_for_best: str = None,
        mode: str = 'max'
) -> Dict[str, Any]:
    """
    Generate comprehensive visualizations for a single fold.

    Args:
        fold_data: Dictionary with fold ge_training history
        fold_idx: Fold index number
        output_dir: Directory to save visualizations
        config: Configuration dictionary
        metric_for_best: Metric to use for determining best epoch
        mode: 'max' to maximize metric, 'min' to minimize

    Returns:
        Dictionary with best epoch metrics
    """
    # Create directories
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    ensure_directory(fold_dir)

    # Determine task type and metrics to use
    is_survival = config['execution'].get('task', 'classification') == 'survival'

    # Set defaults based on task type if not provided
    if metric_for_best is None:
        metric_for_best = 'c_index' if is_survival else 'acc'
    if mode is None:
        mode = 'max'  # Both c-index and accuracy are better when higher

    # Get best epoch and its metrics
    best_epoch, best_metrics = get_best_epoch_data(fold_data['history'], metric_for_best, mode)

    # Get appropriate metrics list based on task
    metrics_to_plot = ['loss', 'c_index'] if is_survival else ['loss', 'acc']

    # 1. Plot learning curves with best epoch marked
    plot_learning_curves(
        fold_data['history'],
        fold_dir,
        metrics=metrics_to_plot,
        best_epoch=best_epoch,
        title_prefix=f"Fold {fold_idx} "
    )

    # Create directory for best epoch results
    best_dir = os.path.join(fold_dir, f"best_epoch_{best_epoch}")
    ensure_directory(best_dir)

    # 2. Get confusion matrix for best epoch

    # Get class names

    if not is_survival and 'confusion_matrix' in fold_data['history']['val']:
        class_names = [config['label_dict'].get(str(i), f"Class {i}") for i in range(config['num_classes'])]

        if ('confusion_matrix' in fold_data['history']['val'] and
                isinstance(fold_data['history']['val']['confusion_matrix'], dict) and
                best_epoch in fold_data['history']['val']['confusion_matrix']):
            cm = fold_data['history']['val']['confusion_matrix'][best_epoch]

            # Plot confusion matrices
            plot_confusion_matrix(
                cm, class_names,
                os.path.join(best_dir, 'confusion_matrix.png'),
                title=f"Fold {fold_idx} Confusion Matrix (Epoch {best_epoch})"
            )

            plot_confusion_matrix(
                cm, class_names,
                os.path.join(best_dir, 'confusion_matrix_normalized.png'),
                title=f"Fold {fold_idx} Normalized Confusion Matrix (Epoch {best_epoch})",
                normalize=True
            )

        # 3. Generate ROC and PR curves for best epoch
        if ('all_labels' in fold_data['history']['val'] and
                isinstance(fold_data['history']['val']['all_labels'], dict) and
                best_epoch in fold_data['history']['val']['all_labels'] and
                'all_probs' in fold_data['history']['val'] and
                best_epoch in fold_data['history']['val']['all_probs']):
            y_true = fold_data['history']['val']['all_labels'][best_epoch]
            y_probs = fold_data['history']['val']['all_probs'][best_epoch]

            # Plot ROC curve
            auc_scores = plot_roc_curves(
                y_true, y_probs, class_names,
                os.path.join(best_dir, 'roc_curve.png'),
                title=f"Fold {fold_idx} ROC Curve (Epoch {best_epoch})"
            )

            # Plot PR curve
            ap_scores = plot_precision_recall_curves(
                y_true, y_probs, class_names,
                os.path.join(best_dir, 'pr_curve.png'),
                title=f"Fold {fold_idx} Precision-Recall Curve (Epoch {best_epoch})"
            )

            # Store scores
            best_metrics['val']['auc_scores'] = auc_scores
            best_metrics['val']['ap_scores'] = ap_scores

    # 4. Create a summary text file
    with open(os.path.join(best_dir, 'best_epoch_summary.txt'), 'w') as f:
        f.write(f"Fold {fold_idx} - Best Epoch {best_epoch} Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Validation Metrics:\n")
        for metric, value in best_metrics.get('val', {}).items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nTraining Metrics:\n")
        for metric, value in best_metrics.get('train', {}).items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric}: {value:.4f}\n")

        # If we have AUC scores per class
        if 'auc_scores' in best_metrics.get('val', {}):
            f.write("\nAUC Scores by Class:\n")
            for class_name, score in best_metrics['val']['auc_scores'].items():
                f.write(f"  {class_name}: {score:.4f}\n")

        # If we have AP scores per class
        if 'ap_scores' in best_metrics.get('val', {}):
            f.write("\nAverage Precision Scores by Class:\n")
            for class_name, score in best_metrics['val']['ap_scores'].items():
                f.write(f"  {class_name}: {score:.4f}\n")

    # Create a dictionary with fold summary for later aggregation
    fold_summary = {
        'fold': fold_idx,
        'best_epoch': best_epoch
    }

    # Add all scalar metrics
    for phase in ['train', 'val']:
        for metric, value in best_metrics.get(phase, {}).items():
            if isinstance(value, (int, float)):
                fold_summary[f"{phase}_{metric}"] = value

    # Add survival-specific visualizations for the best epoch
    if is_survival:
        history_val = fold_data['history']['val']
        if ('all_survival_times' in history_val and
            isinstance(history_val['all_survival_times'], dict) and
            best_epoch in history_val['all_survival_times']):

            survival_times = history_val['all_survival_times'][best_epoch]
            censorships = history_val['all_censorships'][best_epoch]
            risk_scores = history_val['all_risk_scores'][best_epoch]

        survival_dir = os.path.join(fold_dir, f"best_epoch_{best_epoch}", "survival_analysis")
        ensure_directory(survival_dir)

        # Generate survival visualizations
        survival_results = visualize_survival_results(
            survival_times,
            censorships,
            risk_scores,
            survival_dir,
            title_prefix=f"Fold {fold_idx} - ",
            num_groups=2  # Use 2 groups for clearer visualization
        )

        # Add to fold summary
        if 'survival_results' not in fold_summary:
            fold_summary['survival_results'] = {}
        fold_summary['survival_results'] = survival_results

    return fold_summary


def visualize_aggregated_results(
        fold_summaries: List[Dict[str, Any]],
        fold_histories: List[Dict[str, Any]],
        output_dir: str,
        config: Dict[str, Any]
) -> None:
    """
    Generate aggregated visualizations across all folds.

    Args:
        fold_summaries: List of dictionaries with fold summary metrics
        fold_histories: List of dictionaries with full fold histories
        output_dir: Directory to save visualizations
        config: Configuration dictionary
    """
    # Create a DataFrame from fold summaries
    fold_results_df = pd.DataFrame(fold_summaries)

    # Save fold metrics to CSV
    metrics_path = os.path.join(output_dir, 'fold_metrics.csv')
    fold_results_df.to_csv(metrics_path, index=False)
    is_survival = config['execution'].get('task', 'classification') == 'survival'
    metrics = ['loss', 'c_index'] if is_survival else ['loss', 'acc']

    # Generate aggregated plots

    # 1. Plot mean learning curves with std bands
    plot_aggregated_learning_curves(
        [h['history'] for h in fold_histories],
        output_dir,
        metrics=metrics,
        best_epochs=fold_results_df['best_epoch'].tolist()
    )

    # 2. Plot metric comparisons across folds
    plot_metric_comparison_by_fold(fold_results_df, output_dir)

    # 3. Create aggregated confusion matrix if available
    all_cms = []
    for fold_idx, fold_data in enumerate(fold_histories):
        best_epoch = fold_results_df.loc[fold_results_df['fold'] == fold_idx, 'best_epoch'].iloc[0]
        if ('confusion_matrix' in fold_data['history']['val'] and
                isinstance(fold_data['history']['val']['confusion_matrix'], dict) and
                best_epoch in fold_data['history']['val']['confusion_matrix']):
            all_cms.append(fold_data['history']['val']['confusion_matrix'][best_epoch])

    if all_cms:
        # Calculate mean confusion matrix
        avg_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)

        # Get class names
        class_names = [config['label_dict'].get(str(i), f"Class {i}") for i in range(config['num_classes'])]

        # Plot average confusion matrices
        plot_confusion_matrix(
            avg_cm, class_names,
            os.path.join(output_dir, 'average_confusion_matrix.png'),
            title=f"Average Confusion Matrix ({len(all_cms)} folds)"
        )

        plot_confusion_matrix(
            avg_cm, class_names,
            os.path.join(output_dir, 'average_confusion_matrix_normalized.png'),
            title=f"Average Normalized Confusion Matrix ({len(all_cms)} folds)",
            normalize=True
        )

    # 4. Create aggregated ROC and PR curves if possible
    all_labels = []
    all_probs = []

    for fold_idx, fold_data in enumerate(fold_histories):
        best_epoch = fold_results_df.loc[fold_results_df['fold'] == fold_idx, 'best_epoch'].iloc[0]
        if ('all_labels' in fold_data['history']['val'] and
                isinstance(fold_data['history']['val']['all_labels'], dict) and
                best_epoch in fold_data['history']['val']['all_labels'] and
                'all_probs' in fold_data['history']['val'] and
                best_epoch in fold_data['history']['val']['all_probs']):
            all_labels.append(fold_data['history']['val']['all_labels'][best_epoch])
            all_probs.append(fold_data['history']['val']['all_probs'][best_epoch])

    if all_labels and all_probs:
        # Concatenate all predictions
        y_true = np.concatenate(all_labels)
        y_probs = np.concatenate(all_probs)

        # Get class names
        class_names = [config['label_dict'].get(str(i), f"Class {i}") for i in range(config['num_classes'])]

        # Plot ROC curve
        auc_scores = plot_roc_curves(
            y_true, y_probs, class_names,
            os.path.join(output_dir, 'aggregated_roc_curve.png'),
            title=f"Aggregated ROC Curve ({len(all_labels)} folds)"
        )

        # Plot PR curve
        ap_scores = plot_precision_recall_curves(
            y_true, y_probs, class_names,
            os.path.join(output_dir, 'aggregated_pr_curve.png'),
            title=f"Aggregated Precision-Recall Curve ({len(all_labels)} folds)"
        )

    # 5. Create comprehensive summary report
    summary_dir = os.path.join(output_dir, 'summary')
    ensure_directory(summary_dir)
    create_result_summary(fold_results_df, config, summary_dir)


def visualize_full_training_results(
        history: Dict[str, Any],
        output_dir: str,
        config: Dict[str, Any],
        metric_for_best: str = 'acc',
        mode: str = 'max'
) -> None:
    """
    Generate comprehensive visualizations for full ge_training results.

    Args:
        history: Dictionary with ge_training history
        output_dir: Directory to save visualizations
        config: Configuration dictionary
        metric_for_best: Metric to use for determining best epoch
        mode: 'max' to maximize metric, 'min' to minimize
    """
    # Create directories
    ensure_directory(output_dir)

    # Get best epoch and its metrics
    best_epoch, best_metrics = get_best_epoch_data(history, metric_for_best, mode)

    # Get class names
    class_names = [config['label_dict'].get(str(i), f"Class {i}") for i in range(config['num_classes'])]

    is_survival = config['execution'].get('task', 'classification') == 'survival'
    # Get appropriate metrics list based on task
    metrics_to_plot = ['loss', 'c_index'] if is_survival else ['loss', 'acc']

    # 1. Plot learning curves with best epoch marked
    plot_learning_curves(
        history,
        output_dir,
        metrics=metrics_to_plot,
        best_epoch=best_epoch,
        title_prefix="Full Training "
    )

    # Create directory for best epoch results
    best_dir = os.path.join(output_dir, f"best_epoch_{best_epoch}")
    ensure_directory(best_dir)

    # 2. Get confusion matrix for best epoch
    if ('confusion_matrix' in history['val'] and
            isinstance(history['val']['confusion_matrix'], dict) and
            best_epoch in history['val']['confusion_matrix']):
        cm = history['val']['confusion_matrix'][best_epoch]

        # Plot confusion matrices
        plot_confusion_matrix(
            cm, class_names,
            os.path.join(best_dir, 'confusion_matrix.png'),
            title=f"Confusion Matrix (Epoch {best_epoch})"
        )

        plot_confusion_matrix(
            cm, class_names,
            os.path.join(best_dir, 'confusion_matrix_normalized.png'),
            title=f"Normalized Confusion Matrix (Epoch {best_epoch})",
            normalize=True
        )

    # 3. Generate ROC and PR curves for best epoch
    if ('all_labels' in history['val'] and
            isinstance(history['val']['all_labels'], dict) and
            best_epoch in history['val']['all_labels'] and
            'all_probs' in history['val'] and
            best_epoch in history['val']['all_probs']):
        y_true = history['val']['all_labels'][best_epoch]
        y_probs = history['val']['all_probs'][best_epoch]

        # Plot ROC curve
        auc_scores = plot_roc_curves(
            y_true, y_probs, class_names,
            os.path.join(best_dir, 'roc_curve.png'),
            title=f"ROC Curve (Epoch {best_epoch})"
        )

        # Plot PR curve
        ap_scores = plot_precision_recall_curves(
            y_true, y_probs, class_names,
            os.path.join(best_dir, 'pr_curve.png'),
            title=f"Precision-Recall Curve (Epoch {best_epoch})"
        )

        # Store scores
        best_metrics['val']['auc_scores'] = auc_scores
        best_metrics['val']['ap_scores'] = ap_scores

    # 4. Create a summary text file
    with open(os.path.join(best_dir, 'best_epoch_summary.txt'), 'w') as f:
        f.write(f"Full Training - Best Epoch {best_epoch} Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Validation Metrics:\n")
        for metric, value in best_metrics.get('val', {}).items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric}: {value:.4f}\n")

        f.write("\nTraining Metrics:\n")
        for metric, value in best_metrics.get('train', {}).items():
            if isinstance(value, (int, float)):
                f.write(f"  {metric}: {value:.4f}\n")

        # If we have AUC scores per class
        if 'auc_scores' in best_metrics.get('val', {}):
            f.write("\nAUC Scores by Class:\n")
            for class_name, score in best_metrics['val']['auc_scores'].items():
                f.write(f"  {class_name}: {score:.4f}\n")

        # If we have AP scores per class
        if 'ap_scores' in best_metrics.get('val', {}):
            f.write("\nAverage Precision Scores by Class:\n")
            for class_name, score in best_metrics['val']['ap_scores'].items():
                f.write(f"  {class_name}: {score:.4f}\n")

    # Create a summary DataFrame
    summary = {
        'best_epoch': best_epoch
    }

    # Add all scalar metrics
    for phase in ['train', 'val']:
        for metric, value in best_metrics.get(phase, {}).items():
            if isinstance(value, (int, float)):
                summary[f"{phase}_{metric}"] = value

    # Convert to DataFrame for summary generation
    results_df = pd.DataFrame([summary])

    # 5. Create comprehensive summary report
    summary_dir = os.path.join(output_dir, 'summary')
    ensure_directory(summary_dir)
    create_result_summary(results_df, config, summary_dir)

    # Save summary as CSV
    results_df.to_csv(os.path.join(summary_dir, 'full_training_metrics.csv'), index=False)