import os
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import (precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, average_precision_score)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, test_loader, config, device):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            target = batch.y
            outputs = model(batch)

            # Calculate loss just for reference
            loss = F.cross_entropy(outputs, target)
            pred = outputs.argmax(dim=1)
            probs = F.softmax(outputs, dim=1)

            total_loss += loss.item()
            correct += (pred == target).sum().item()
            total += target.size(0)

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    avg_acc = 100. * correct / total

    metrics = {
        'loss': avg_loss,
        'acc': avg_acc,
        'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'classification_report': classification_report(all_targets, all_preds, zero_division=0, output_dict=True),
        'all_labels': all_targets,
        'all_preds': all_preds,
        'all_probs': all_probs
    }

    if all_probs.shape[1] == 2:
        metrics['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
    else:
        n_classes = config['n_classes']
        binary_labels = label_binarize(all_targets, classes=range(n_classes))
        metrics['auc'] = roc_auc_score(binary_labels, all_probs, average='macro', multi_class='ovr')
        metrics['precision'] = average_precision_score(all_targets,
                                                       label_binarize(all_preds, classes=n_classes),
                                                       average='macro')

    return metrics

def save_metrics(test_dataset, test_metrics, test_results_dir, config, logger):

    # Extract metrics
    accuracy = test_metrics['acc'] / 100.0  # Convert from percentage
    precision = test_metrics['precision']
    recall = test_metrics['recall']
    f1 = test_metrics['f1']
    auc = test_metrics.get('auc', 0.0)
    conf_matrix = test_metrics['confusion_matrix']
    class_report = test_metrics['classification_report']
    all_preds = test_metrics['all_preds']
    all_targets = test_metrics['all_labels']
    all_probs = test_metrics['all_probs']

    # Prepare results dictionary
    test_results = {
        'dataset': config['dataset_name'],
        'model': config['model']['name'],
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
        },
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }

    # Also save as human-readable text
    with open(os.path.join(test_results_dir, 'test_results.txt'), 'w') as f:
        f.write("=== ProtoPathway Test Results ===\n\n")
        f.write(f"Dataset: {config['dataset_name']}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_targets, all_preds, zero_division=0))

    logger.info(f"Test results saved to {test_results_dir}/test_results.txt")
    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # Store patient-level predictions for further analysis
    patient_ids = []
    for idx in range(len(test_dataset)):
        if config['model']['name'] == 'MLP':
            patient_ids.append(test_dataset[idx]['id'])
        elif config['model']['name'] == 'Hypergraph':
            patient_ids.append(test_dataset[idx].patient_id)

    # Ensure we have the right number of predictions
    assert len(patient_ids) == len(all_targets), "Number of patients doesn't match number of predictions"

    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': all_targets,
        'predicted_label': all_preds
    })

    # Add probability columns
    for i in range(all_probs.shape[1]):
        class_name = config['label_dict'].get(str(i), f'class_{i}')
        pred_df[f'prob_{class_name}'] = all_probs[:, i]

    # Save prediction dataframe
    pred_df.to_csv(os.path.join(test_results_dir, 'patient_predictions.csv'), index=False)

    return test_results