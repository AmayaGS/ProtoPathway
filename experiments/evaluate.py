"""
Evaluation Script for ProtoPathway.

Loads trained model checkpoints and performs:
- Comprehensive metric computation
- Attention weight extraction for visualization
- Patient-level prediction export
- Risk stratification analysis (survival)
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from utils.dataset import load_dataset_components, MultimodalDataset
from utils.survival import calculate_risk
# from utils.losses import NLLSurvLoss

# Use factory instead of direct import
from models.factory import build_model, get_model_requirements


@torch.no_grad()
def evaluate_model(model, loader, cfg, device, return_attention=False):
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        loader: DataLoader for evaluation data
        cfg: Config object
        device: Computation device
        return_attention: Whether to extract attention weights

    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()

    all_patient_ids = []
    all_logits = []

    # Survival specific
    all_risks = []
    all_times = []
    all_events = []
    all_bins = []

    # Classification specific
    all_probs = []
    all_targets = []

    # Attention weights
    attention_outputs = {} if return_attention else None

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch, return_attention=return_attention)

        patient_id = batch.patient_id if hasattr(batch, 'patient_id') else None

        if cfg.task == 'survival':
            risk = calculate_risk(logits)
            all_risks.append(risk.cpu())
            all_times.append(batch.y['time'].cpu())
            all_events.append(batch.y['event'].cpu())
            all_bins.append(batch.y['bin'].cpu())
        else:
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())

        all_logits.append(logits.cpu())
        if patient_id is not None:
            all_patient_ids.append(patient_id)

        # Collect attention weights
        if return_attention and hasattr(model, 'get_attention_outputs'):
            attn = model.get_attention_outputs()
            for key, value in attn.items():
                if value is not None:
                    if key not in attention_outputs:
                        attention_outputs[key] = []
                    # Move tensors to CPU and detach
                    if isinstance(value, torch.Tensor):
                        attention_outputs[key].append(value.cpu())
                    elif isinstance(value, dict):
                        attention_outputs[key].append({
                            k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in value.items()
                        })

    # Flatten patient IDs
    patient_ids = []
    for pid in all_patient_ids:
        if isinstance(pid, (list, tuple)):
            patient_ids.extend(pid)
        else:
            patient_ids.append(pid)

    results = {'patient_ids': patient_ids}

    if cfg.task == 'survival':
        risks = torch.cat(all_risks).numpy()
        times = torch.cat(all_times).numpy()
        events = torch.cat(all_events).numpy()
        bins = torch.cat(all_bins).numpy()

        # C-index
        event_indicator = events.astype(bool)
        try:
            c_index = concordance_index_censored(event_indicator, times, risks)[0]
        except Exception as e:
            logging.warning(f"Could not compute c-index: {e}")
            c_index = 0.5

        results.update({
            'c_index': c_index,
            'risks': risks,
            'times': times,
            'events': events,
            'bins': bins,
            'n_events': int(event_indicator.sum()),
            'n_censored': int((~event_indicator).sum()),
            'event_rate': float(event_indicator.mean())
        })

    else:
        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()
        preds = probs.argmax(axis=1)

        # Metrics
        accuracy = accuracy_score(targets, preds) * 100
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        recall = recall_score(targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)

        # AUC
        if probs.shape[1] == 2:
            auc = roc_auc_score(targets, probs[:, 1])
        else:
            try:
                auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            except:
                auc = 0.5

        conf_matrix = confusion_matrix(targets, preds)
        class_report = classification_report(targets, preds, output_dict=True, zero_division=0)

        results.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'probs': probs,
            'targets': targets,
            'preds': preds
        })

    if return_attention:
        results['attention_outputs'] = attention_outputs

    return results


def save_predictions(results, output_dir, cfg, fold_idx=None):
    """Save patient-level predictions to CSV."""
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""

    if cfg.task == 'survival':
        df = pd.DataFrame({
            'patient_id': results['patient_ids'],
            'risk_score': results['risks'],
            'survival_time': results['times'],
            'event': results['events'],
            'survival_bin': results['bins']
        })
    else:
        df = pd.DataFrame({
            'patient_id': results['patient_ids'],
            'predicted_class': results['preds'],
            'true_class': results['targets']
        })
        # Add probability columns
        for i in range(results['probs'].shape[1]):
            df[f'prob_class_{i}'] = results['probs'][:, i]

    csv_path = os.path.join(output_dir, f'predictions{suffix}.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved predictions to {csv_path}")

    return csv_path


def save_attention_weights(attention_outputs, patient_ids, output_dir, fold_idx=None):
    """Save attention weights for visualization."""
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    attention_dir = os.path.join(output_dir, f'attention{suffix}')
    os.makedirs(attention_dir, exist_ok=True)

    import pickle

    attention_data = {
        'patient_ids': patient_ids,
        'attention_outputs': attention_outputs
    }

    pkl_path = os.path.join(attention_dir, 'attention_weights.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(attention_data, f)

    logging.info(f"Saved attention weights to {pkl_path}")

    return pkl_path


def generate_metrics_report(results, output_path, cfg):
    """Generate a human-readable metrics report."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ProtoPathway Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset: {cfg.dataset}\n")
        f.write(f"Task: {cfg.task}\n")
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Number of patients: {len(results['patient_ids'])}\n\n")

        if cfg.task == 'survival':
            f.write("--- Survival Metrics ---\n")
            f.write(f"C-index: {results['c_index']:.4f}\n")
            f.write(f"Events: {results['n_events']}\n")
            f.write(f"Censored: {results['n_censored']}\n")
            f.write(f"Event rate: {results['event_rate']:.2%}\n")
        else:
            f.write("--- Classification Metrics ---\n")
            f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 Score: {results['f1']:.4f}\n")
            f.write(f"AUC: {results['auc']:.4f}\n\n")

            f.write("Confusion Matrix:\n")
            for row in results['confusion_matrix']:
                f.write(f"  {row}\n")

    logging.info(f"Saved metrics report to {output_path}")


def run(cfg):
    """
    Main evaluation entry point.
    Iterates all fold checkpoints, evaluates each, saves per-fold outputs,
    and computes aggregate CV metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    exp_dir = Path(cfg.checkpoint_dir)

    # Find all fold checkpoints
    checkpoint_paths = sorted(exp_dir.glob('best_model_fold_*.pt'))
    if not checkpoint_paths:
        checkpoint_paths = sorted(exp_dir.glob('*fold*.pt'))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {exp_dir}")

    logging.info(f"Found {len(checkpoint_paths)} fold checkpoints")

    # Load data components once (shared across folds)
    data_components = load_dataset_components(cfg)
    splits = data_components['splits']

    return_attention = True

    output_dir = exp_dir / 'evaluation'
    os.makedirs(output_dir, exist_ok=True)

    fold_results = []

    for ckpt_path in checkpoint_paths:
        # Parse fold index from filename
        fold_idx = int(ckpt_path.stem.split('_')[-1])
        fold_name = f"Fold {fold_idx}"
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating {fold_name}")
        logging.info(f"Checkpoint: {ckpt_path}")
        logging.info(f"{'='*60}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Get validation patients for this fold
        patient_ids = splits['CV'][fold_name]['Val']
        logging.info(f"  Patients: {len(patient_ids)}")

        # Create dataset and loader
        eval_dataset = MultimodalDataset(
            patient_ids=patient_ids,
            gene_expression_df=data_components['gene_expression_df'],
            graph_data=data_components['graph_data'],
            wsi_features=data_components['wsi_features'],
            labels_df=data_components['labels_df'],
            task=cfg.task,
            patient_id_col=cfg.patient_id_col
        )

        eval_loader = PyGDataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers
        )

        # Build model
        model_kwargs = {
            'num_genes': data_components['graph_data']['num_genes'],
            'num_pathways': data_components['graph_data']['num_pathways']
        }

        model_reqs = get_model_requirements(cfg.model.name)
        if model_reqs.get('needs_centroids') and cfg.model.get('branches', {}).get('wsi', False):
            centroid_path = exp_dir / f'centroids_fold_{fold_idx}.pt'
            if centroid_path.exists():
                model_kwargs['wsi_centroids'] = torch.load(centroid_path, weights_only=True)
                logging.info("  Loaded pre-computed centroids")

        model = build_model(cfg, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        logging.info(f"  Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

        # Evaluate
        results = evaluate_model(model, eval_loader, cfg, device, return_attention=return_attention)

        # Log fold summary
        if cfg.task == 'survival':
            logging.info(f"  C-index: {results['c_index']:.4f}")
            logging.info(f"  Events: {results['n_events']}, Censored: {results['n_censored']}")
        else:
            logging.info(f"  Accuracy: {results['accuracy']:.2f}%")
            logging.info(f"  AUC: {results['auc']:.4f}")

        # Save per-fold outputs
        save_predictions(results, output_dir, cfg, fold_idx=fold_idx)
        generate_metrics_report(results, output_dir / f'metrics_report_fold_{fold_idx}.txt', cfg)

        if return_attention and 'attention_outputs' in results:
            save_attention_weights(
                results['attention_outputs'],
                results['patient_ids'],
                output_dir,
                fold_idx=fold_idx
            )

        # Store for aggregation
        fold_results.append({'fold_idx': fold_idx, **results})

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ---- Aggregate CV metrics ----
    logging.info(f"\n{'='*60}")
    logging.info("Cross-Validation Summary")
    logging.info(f"{'='*60}")

    cv_summary = aggregate_cv_results(fold_results, cfg)

    json_path = output_dir / 'cv_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)

    report_path = output_dir / 'cv_summary.txt'
    write_cv_summary_report(cv_summary, report_path, cfg)

    logging.info(f"\nAll results saved to {output_dir}")

    return fold_results, cv_summary


def aggregate_cv_results(fold_results, cfg):
    """Compute mean ± std across folds."""
    summary = {'n_folds': len(fold_results)}

    if cfg.task == 'survival':
        c_indices = [r['c_index'] for r in fold_results]
        event_rates = [r['event_rate'] for r in fold_results]

        summary.update({
            'mean_c_index': float(np.mean(c_indices)),
            'std_c_index': float(np.std(c_indices)),
            'per_fold': {
                f"fold_{r['fold_idx']}": {
                    'c_index': r['c_index'],
                    'n_events': r['n_events'],
                    'n_censored': r['n_censored'],
                    'event_rate': r['event_rate'],
                    'n_patients': r['n_events'] + r['n_censored']
                }
                for r in fold_results
            },
            'mean_event_rate': float(np.mean(event_rates))
        })

        logging.info(f"  C-index: {summary['mean_c_index']:.4f} ± {summary['std_c_index']:.4f}")
        for fold_key, fold_data in summary['per_fold'].items():
            logging.info(f"    {fold_key}: {fold_data['c_index']:.4f} "
                        f"(n={fold_data['n_patients']}, events={fold_data['n_events']})")
    else:
        metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        for m in metrics:
            vals = [r[m] for r in fold_results]
            summary[f'mean_{m}'] = float(np.mean(vals))
            summary[f'std_{m}'] = float(np.std(vals))

        summary['per_fold'] = {
            f"fold_{r['fold_idx']}": {m: r[m] for m in metrics}
            for r in fold_results
        }

        logging.info(f"  Accuracy: {summary['mean_accuracy']:.2f} ± {summary['std_accuracy']:.2f}")
        logging.info(f"  AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
        logging.info(f"  F1:  {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    return summary


def write_cv_summary_report(cv_summary, output_path, cfg):
    """Write human-readable CV summary."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ProtoPathway Cross-Validation Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset: {cfg.dataset}\n")
        f.write(f"Task: {cfg.task}\n")
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Number of folds: {cv_summary['n_folds']}\n\n")

        if cfg.task == 'survival':
            f.write(f"C-index: {cv_summary['mean_c_index']:.4f} ± {cv_summary['std_c_index']:.4f}\n\n")
            f.write("Per-fold breakdown:\n")
            for fold_key, fold_data in cv_summary['per_fold'].items():
                f.write(f"  {fold_key}: C-index={fold_data['c_index']:.4f}  "
                       f"n={fold_data['n_patients']}  "
                       f"events={fold_data['n_events']}  "
                       f"censored={fold_data['n_censored']}\n")
        else:
            f.write(f"Accuracy:  {cv_summary['mean_accuracy']:.2f} ± {cv_summary['std_accuracy']:.2f}\n")
            f.write(f"AUC:       {cv_summary['mean_auc']:.4f} ± {cv_summary['std_auc']:.4f}\n")
            f.write(f"F1:        {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}\n")
            f.write(f"Precision: {cv_summary['mean_precision']:.4f} ± {cv_summary['std_precision']:.4f}\n")
            f.write(f"Recall:    {cv_summary['mean_recall']:.4f} ± {cv_summary['std_recall']:.4f}\n\n")
            f.write("Per-fold breakdown:\n")
            for fold_key, fold_data in cv_summary['per_fold'].items():
                f.write(f"  {fold_key}: Acc={fold_data['accuracy']:.2f}  "
                       f"AUC={fold_data['auc']:.4f}  F1={fold_data['f1']:.4f}\n")

    logging.info(f"Saved CV summary to {output_path}")