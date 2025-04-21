
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import itertools
import logging

from utils.helpers import ensure_directory

logger = logging.getLogger('protopathway')


def plot_confusion_matrix(cm, classes, output_path, title='Confusion Matrix',
                          normalize=False, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        classes: List of class names
        output_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add labels to each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_roc_curves(all_labels, all_probs, classes, output_path):
    """
    Plot ROC curves for multi-class classification.

    Args:
        all_labels: Ground truth labels
        all_probs: Predicted probabilities
        classes: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Binarize the labels for one-vs-rest ROC
    y_test_bin = np.zeros((len(all_labels), len(classes)))
    for i in range(len(all_labels)):
        y_test_bin[i, all_labels[i]] = 1

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(classes)))

    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return roc_auc


def plot_precision_recall_curves(all_labels, all_probs, classes, output_path):
    """
    Plot precision-recall curves for multi-class classification.

    Args:
        all_labels: Ground truth labels
        all_probs: Predicted probabilities
        classes: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()

    # Binarize the labels for one-vs-rest PR curves
    y_test_bin = np.zeros((len(all_labels), len(classes)))
    for i in range(len(all_labels)):
        y_test_bin[i, all_labels[i]] = 1

    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        avg_precision[i] = np.mean(precision[i])

    # Plot all precision-recall curves
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(classes)))

    for i, color in zip(range(len(classes)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of class {classes[i]} (AP = {avg_precision[i]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curves')
    plt.legend(loc="best")

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return avg_precision


def plot_learning_curves(history, output_path, metrics=None):
    """
    Plot learning curves from training history.

    Args:
        history: Dictionary containing training history
        output_path: Path to save the plots
        metrics: List of metrics to plot (defaults to all)
    """
    if not metrics:
        # Get all available metrics excluding non-numeric or special ones
        metrics = set(list(history['train'].keys()) + list(history['val'].keys()))
        # Remove special metrics
        metrics = [m for m in metrics if m not in ['confusion_matrix', 'classification_report']]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Plot training & validation metrics
        if metric in history['train']:
            train_values = history['train'][metric]
            epochs = range(1, len(train_values) + 1)
            plt.plot(epochs, train_values, 'b-', label=f'Training {metric}')

        if metric in history['val']:
            val_values = history['val'][metric]
            epochs = range(1, len(val_values) + 1)
            plt.plot(epochs, val_values, 'r-', label=f'Validation {metric}')

        plt.title(f'{metric.capitalize()} vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Ensure x-axis shows integers for epochs
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save plot
        plt.savefig(os.path.join(output_path, f'{metric}_curve.png'), bbox_inches='tight')
        plt.close()


def plot_fold_comparison(fold_results, output_path, config):
    """
    Plot comparison of metrics across folds.

    Args:
        fold_results: DataFrame containing results for each fold
        output_path: Path to save the plots
        config: Configuration dictionary
    """
    # Get metrics to plot (exclude non-numeric columns)
    metrics = [col for col in fold_results.columns if col not in ['fold', 'best_epoch']]

    # Plot bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Create bar chart
        sns.barplot(x='fold', y=metric, data=fold_results)

        # Add labels and title
        plt.title(f'{metric.capitalize()} by Fold')
        plt.xlabel('Fold')
        plt.ylabel(metric.capitalize())

        # Add mean line
        mean_value = fold_results[metric].mean()
        plt.axhline(y=mean_value, color='r', linestyle='--',
                    label=f'Mean: {mean_value:.4f}')

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(os.path.join(output_path, f'{metric}_by_fold.png'), bbox_inches='tight')
        plt.close()

    # Create a heatmap of all metrics across folds
    plt.figure(figsize=(12, 8))

    # Pivot the data for heatmap
    heatmap_data = fold_results.copy()
    heatmap_data['fold'] = heatmap_data['fold'].apply(lambda x: f'Fold {x}')
    heatmap_data = heatmap_data.set_index('fold')

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('Metrics by Fold')
    plt.tight_layout()

    # Save heatmap
    plt.savefig(os.path.join(output_path, 'fold_metrics_heatmap.png'), bbox_inches='tight')
    plt.close()


def visualize_model_results(all_fold_results, config, output_dir):
    """
    Create comprehensive visualizations of model results.

    Args:
        all_fold_results: Dictionary containing results from all folds
        config: Configuration dictionary
        output_dir: Directory to save visualizations
    """
    plots_dir = os.path.join(output_dir, 'plots')
    ensure_directory(plots_dir)

    # Extract necessary data
    fold_metrics = []
    for i, fold_history in enumerate(all_fold_results):
        fold_data = {
            'fold': i,
            'train_loss': fold_history['history']['train']['loss'][-1],
            'train_acc': fold_history['history']['train']['acc'][-1],
            'val_loss': fold_history['history']['val']['loss'][-1],
            'val_acc': fold_history['history']['val']['acc'][-1],
            'best_epoch': fold_history.get('best_epoch', len(fold_history['history']['train']['loss']))
        }

        # Add any additional metrics
        for phase in ['train', 'val']:
            for metric, values in fold_history['history'][phase].items():
                if metric not in ['loss', 'acc'] and len(values) > 0:
                    fold_data[f'{phase}_{metric}'] = values[-1]

        fold_metrics.append(fold_data)

    # Convert to DataFrame
    fold_results_df = pd.DataFrame(fold_metrics)

    # Save fold metrics to CSV
    metrics_path = os.path.join(output_dir, 'fold_metrics.csv')
    fold_results_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved fold metrics to {metrics_path}")

    # Plot fold comparisons
    plot_fold_comparison(fold_results_df, plots_dir, config)

    # Plot learning curves for each fold
    for i, fold_history in enumerate(all_fold_results):
        fold_plots_dir = os.path.join(plots_dir, f'fold_{i}')
        ensure_directory(fold_plots_dir)
        plot_learning_curves(fold_history['history'], fold_plots_dir)

    logger.info(f"Saved visualizations to {plots_dir}")

    return fold_results_df


def create_result_summary(results_df, config, output_path):
    """
    Create a summary of results.

    Args:
        results_df: DataFrame containing fold metrics
        config: Configuration dictionary
        output_path: Path to save the summary
    """
    # Calculate mean and std of metrics
    metrics_summary = results_df.describe().loc[['mean', 'std', 'min', 'max']]

    # Save summary to CSV
    metrics_summary.to_csv(os.path.join(output_path, 'metrics_summary.csv'))

    # Create text summary
    with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
        f.write("ProtoPathway Model Results Summary\n")
        f.write("==================================\n\n")

        f.write(f"Dataset: {config['dataset_name']}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Number of folds: {len(results_df)}\n\n")

        f.write("Metrics Summary:\n")
        for metric in results_df.columns:
            if metric != 'fold' and metric != 'best_epoch':
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                f.write(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}\n")

        f.write(f"\nAverage best epoch: {results_df['best_epoch'].mean():.1f}\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Learning rate: {config['training']['learning_rate']}\n")
        f.write(f"  Batch size: {config['training']['batch_size']}\n")
        f.write(f"  Number of epochs: {config['training']['num_epochs']}\n")
        f.write(f"  Dropout rate: {config['training']['dropout_rate']}\n")
        f.write(f"  L1 regularization: {config['training']['L1_norm']}\n")
        f.write(f"  L2 regularization: {config['training']['L2_norm']}\n")

    logger.info(f"Created results summary at {output_path}")