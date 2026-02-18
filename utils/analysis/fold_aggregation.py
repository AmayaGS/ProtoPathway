"""
Cross-Validation Fold Aggregation.

Pools predictions and attention data across CV folds into unified
patient-level datasets. Each patient appears in exactly one validation
fold, so pooling produces the full cohort with no overlap.

Expected directory structure from evaluate.py:
    evaluation/
        predictions_fold_0.csv
        predictions_fold_1.csv
        ...
        attention_fold_0/attention_weights.pkl
        attention_fold_1/attention_weights.pkl
        ...
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def pool_cv_results(
    eval_dir: str,
    risk_stratification: str = 'median'
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Pool predictions and attention data across all CV folds.

    Args:
        eval_dir: Path to evaluation directory containing per-fold outputs.
        risk_stratification: How to split into risk groups.
            'median': median split (2 groups)
            'quartile': quartile split (4 groups)

    Returns:
        predictions: DataFrame with columns [patient_id, risk_score, survival_time,
                     event, survival_bin, fold, risk_group]
        attention_by_patient: Dict mapping patient_id → {
            'gene_pathway_attention': Tensor [G, P],
            'pathway_importance': Tensor [P],
            'patch_assignments': dict with soft/hard assignments and gate_weights,
            'cross_modal_attention': Tensor [N, P],
            'fusion_gate_weights': Tensor [N],
            'fold': int
        }
        metadata: Dict with 'gene_names', 'pathway_names', 'num_prototypes', etc.
    """
    eval_dir = Path(eval_dir)

    # --- Pool predictions ---
    predictions = _pool_predictions(eval_dir)

    # --- Assign risk groups ---
    predictions = _assign_risk_groups(predictions, risk_stratification)

    # --- Pool attention data ---
    attention_by_patient, metadata = _pool_attention(eval_dir)

    # --- Cross-reference: ensure attention patients match prediction patients ---
    pred_patients = set(predictions['patient_id'].values)
    attn_patients = set(attention_by_patient.keys())

    matched = pred_patients & attn_patients
    pred_only = pred_patients - attn_patients
    attn_only = attn_patients - pred_patients

    if pred_only:
        logger.warning(
            f"{len(pred_only)} patients have predictions but no attention data"
        )
    if attn_only:
        logger.warning(
            f"{len(attn_only)} patients have attention data but no predictions"
        )

    # Add risk group to attention data
    risk_map = dict(
        zip(predictions['patient_id'], predictions['risk_group'])
    )
    for pid in attention_by_patient:
        attention_by_patient[pid]['risk_group'] = risk_map.get(pid, None)

    logger.info(
        f"Pooled {len(predictions)} patients across "
        f"{predictions['fold'].nunique()} folds "
        f"({len(matched)} with both predictions and attention)"
    )

    n_high = (predictions['risk_group'] == 'High Risk').sum()
    n_low = (predictions['risk_group'] == 'Low Risk').sum()
    logger.info(f"Risk stratification: {n_low} Low Risk, {n_high} High Risk")

    return predictions, attention_by_patient, metadata


def _pool_predictions(eval_dir: Path) -> pd.DataFrame:
    """Load and concatenate per-fold prediction CSVs."""
    pred_files = sorted(eval_dir.glob('predictions_fold_*.csv'))

    if not pred_files:
        # Try without fold suffix
        pred_files = sorted(eval_dir.glob('predictions*.csv'))

    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found in {eval_dir}"
        )

    dfs = []
    for f in pred_files:
        df = pd.read_csv(f)

        # Extract fold index from filename
        stem = f.stem
        if '_fold_' in stem:
            fold_idx = int(stem.split('_fold_')[-1])
        else:
            fold_idx = len(dfs)
        df['fold'] = fold_idx

        # Normalise risk scores within each fold so that scores from
        # different models (different folds) are on a comparable scale.
        # Rank-percentile is robust to outliers and non-Gaussian shapes.
        if 'risk_score' in df.columns and len(df) > 1:
            raw = df['risk_score'].values
            df['risk_score_raw'] = raw                        # keep original
            df['risk_score'] = pd.Series(raw).rank(pct=True).values

            logger.info(
                f"  Fold {fold_idx}: {len(df)} patients, "
                f"raw risk [{raw.min():.4f}, {raw.max():.4f}] → "
                f"rank-normalised [0, 1]"
            )

        dfs.append(df)

    predictions = pd.concat(dfs, ignore_index=True)

    # Check for duplicate patients (shouldn't happen in proper CV)
    duplicates = predictions['patient_id'].duplicated()
    if duplicates.any():
        n_dup = duplicates.sum()
        logger.warning(
            f"{n_dup} duplicate patient IDs found across folds. "
            f"Keeping first occurrence."
        )
        predictions = predictions.drop_duplicates(
            subset='patient_id', keep='first'
        )

    logger.info(
        f"Loaded predictions: {len(predictions)} patients from "
        f"{len(pred_files)} folds"
    )

    return predictions


# Canonical group names — importable by other modules
GROUP_NAMES_2 = ['Low Risk', 'High Risk']
GROUP_NAMES_4 = ['Very Low Risk', 'Low Risk', 'High Risk', 'Very High Risk']


def stratify_risk_scores(risk_scores: np.ndarray, n_groups: int = 2):
    """
    Assign patients to risk groups.  Single source of truth used by both
    importance analysis and KM plotting.

    Uses np.searchsorted(side='right') so that a value exactly on a
    boundary is placed in the lower group, matching pd.cut's default
    right-inclusive behaviour: (q1, q2] → group 1.

    Args:
        risk_scores: 1-D array of (normalised) risk scores.
        n_groups: 2 (median split) or 4 (quartile split).

    Returns:
        groups: int array of group indices (0 … n_groups-1).
        group_names: list of display labels.
    """
    if n_groups == 2:
        threshold = np.median(risk_scores)
        groups = (risk_scores > threshold).astype(int)
        return groups, GROUP_NAMES_2
    elif n_groups == 4:
        quartiles = np.percentile(risk_scores, [25, 50, 75])
        # searchsorted(side='right'): value == boundary → stays in lower bin
        groups = np.searchsorted(quartiles, risk_scores, side='right')
        groups = np.clip(groups, 0, 3)
        return groups, GROUP_NAMES_4
    else:
        raise ValueError(f"n_groups must be 2 or 4, got {n_groups}")


def _assign_risk_groups(
    predictions: pd.DataFrame,
    stratification: str = 'median'
) -> pd.DataFrame:
    """Assign risk groups based on predicted risk scores."""
    risk_scores = predictions['risk_score'].values

    if stratification == 'median':
        n_groups = 2
    elif stratification == 'quartile':
        n_groups = 4
    else:
        raise ValueError(
            f"Unknown stratification: {stratification}. "
            f"Use 'median' or 'quartile'."
        )

    groups, group_names = stratify_risk_scores(risk_scores, n_groups)
    predictions['risk_group'] = [group_names[g] for g in groups]

    return predictions

def _pool_attention(
    eval_dir: Path
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Load and merge per-fold attention pickles into per-patient dict.

    Each fold's pickle has:
        {
            'patient_ids': [list],
            'attention_outputs': {
                'gene_pathway_attention': [list of tensors],
                'pathway_importance': [list of tensors],
                'patch_assignments': [list of dicts],
                'cross_modal_attention': [list of tensors],
                'fusion_gate_weights': [list of tensors],
            }
        }
    """
    attn_dirs = sorted(eval_dir.glob('attention_fold_*'))

    if not attn_dirs:
        # Try alternate naming
        attn_dirs = sorted(eval_dir.glob('attention*'))

    if not attn_dirs:
        logger.warning("No attention data found")
        return {}, {}

    attention_by_patient = {}
    metadata = {}

    for attn_dir in attn_dirs:
        pkl_path = Path(attn_dir) / 'attention_weights.pkl'
        if not pkl_path.exists():
            # Maybe the path IS the pkl file
            if str(attn_dir).endswith('.pkl'):
                pkl_path = attn_dir
            else:
                logger.warning(f"No attention_weights.pkl in {attn_dir}")
                continue

        # Extract fold index
        dirname = attn_dir.stem if attn_dir.is_dir() else attn_dir.parent.stem
        if '_fold_' in dirname:
            fold_idx = int(dirname.split('_fold_')[-1])
        else:
            fold_idx = len(attention_by_patient)

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        patient_ids = data['patient_ids']
        attn_outputs = data['attention_outputs']

        if not metadata.get('gene_names'):
            if data.get('gene_names') is not None:
                metadata['gene_names'] = data['gene_names']
            if data.get('pathway_names') is not None:
                metadata['pathway_names'] = data['pathway_names']

        # Determine number of patients from the first available key
        n_patients = len(patient_ids)

        # Infer metadata from first patient's data (shapes etc.)
        if not metadata:
            metadata = _extract_metadata(attn_outputs)

        # Unpack per-patient
        for i, pid in enumerate(patient_ids):
            if pid in attention_by_patient:
                logger.warning(
                    f"Patient {pid} appears in multiple folds, "
                    f"keeping first"
                )
                continue

            patient_attn = {'fold': fold_idx}

            # Extract each signal for this patient
            for key, values in attn_outputs.items():
                if i >= len(values):
                    continue

                val = values[i]

                # Convert tensors to numpy for consistency
                if HAS_TORCH and isinstance(val, torch.Tensor):
                    patient_attn[key] = val.numpy()
                elif isinstance(val, dict):
                    patient_attn[key] = {
                        k: v.numpy() if (HAS_TORCH and isinstance(v, torch.Tensor)) else v
                        for k, v in val.items()
                    }
                else:
                    patient_attn[key] = val

            attention_by_patient[pid] = patient_attn

    logger.info(
        f"Loaded attention data for {len(attention_by_patient)} patients "
        f"from {len(attn_dirs)} folds"
    )

    return attention_by_patient, metadata


def _extract_metadata(attn_outputs: Dict) -> Dict:
    """Extract shape metadata from the first patient's attention data."""
    metadata = {}

    if 'gene_pathway_attention' in attn_outputs and attn_outputs['gene_pathway_attention']:
        first = attn_outputs['gene_pathway_attention'][0]
        if hasattr(first, 'shape'):
            metadata['num_genes'] = first.shape[0]
            metadata['num_pathways'] = first.shape[1]

    if 'pathway_importance' in attn_outputs and attn_outputs['pathway_importance']:
        first = attn_outputs['pathway_importance'][0]
        size = first.shape[0] if hasattr(first, 'shape') else len(first)
        metadata['num_pathways'] = size

    if 'cross_modal_attention' in attn_outputs and attn_outputs['cross_modal_attention']:
        first = attn_outputs['cross_modal_attention'][0]
        if hasattr(first, 'shape') and len(first.shape) == 2:
            metadata['num_prototypes'] = first.shape[0]

    if 'patch_assignments' in attn_outputs and attn_outputs['patch_assignments']:
        first = attn_outputs['patch_assignments'][0]
        if isinstance(first, dict) and 'gate_weights' in first:
            gw = first['gate_weights']
            size = gw.shape[0] if hasattr(gw, 'shape') else len(gw)
            metadata['num_prototypes'] = size

    return metadata


def get_entity_names(
    eval_dir: str,
    gene_idx_path: Optional[str] = None,
    pathway_idx_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load gene and pathway names from index files.

    These are typically stored alongside the dataset or in the
    experiment config. This function tries common locations.

    Args:
        eval_dir: Evaluation directory (to search relative paths)
        gene_idx_path: Explicit path to gene index pickle/json
        pathway_idx_path: Explicit path to pathway index pickle/json

    Returns:
        Dict with 'gene_names' and 'pathway_names' lists,
        ordered by index.
    """
    names = {}

    # Try loading from provided paths or common locations
    eval_dir = Path(eval_dir)
    search_dirs = [
        eval_dir,
        eval_dir.parent,
        eval_dir.parent / 'data',
        eval_dir.parent / 'processed'
    ]

    # Gene names
    if gene_idx_path and os.path.exists(gene_idx_path):
        names['gene_names'] = _load_idx_file(gene_idx_path)
    else:
        for d in search_dirs:
            for pattern in ['gene_idx.*', 'gene_names.*', '*gene_index*']:
                matches = list(d.glob(pattern))
                if matches:
                    names['gene_names'] = _load_idx_file(matches[0])
                    break
            if 'gene_names' in names:
                break

    # Pathway names
    if pathway_idx_path and os.path.exists(pathway_idx_path):
        names['pathway_names'] = _load_idx_file(pathway_idx_path)
    else:
        for d in search_dirs:
            for pattern in ['pathway_idx.*', 'pathway_names.*', '*pathway_index*']:
                matches = list(d.glob(pattern))
                if matches:
                    names['pathway_names'] = _load_idx_file(matches[0])
                    break
            if 'pathway_names' in names:
                break

    return names


def _load_idx_file(path: Path) -> List[str]:
    """Load an index mapping file and return names ordered by index."""
    path = Path(path)

    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            idx_dict = pickle.load(f)
    elif path.suffix == '.json':
        import json
        with open(path) as f:
            idx_dict = json.load(f)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
        # Assume first column is name, second is index
        return df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unsupported index file format: {path.suffix}")

    # idx_dict maps name → index (int or tensor)
    if isinstance(idx_dict, dict):
        items = []
        for name, idx in idx_dict.items():
            if HAS_TORCH and isinstance(idx, torch.Tensor):
                idx = idx.item()
            items.append((name, int(idx)))
        items.sort(key=lambda x: x[1])
        return [name for name, _ in items]

    # If it's already a list, return as-is
    if isinstance(idx_dict, list):
        return idx_dict

    raise ValueError(f"Cannot parse index file: {path}")