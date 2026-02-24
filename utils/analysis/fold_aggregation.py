"""
Cross-Validation Fold Aggregation.

Pools predictions and attention data across CV folds into unified
patient-level datasets. Each patient appears in exactly one validation
fold, so pooling produces the full cohort with no overlap.

Includes prototype alignment across folds via Hungarian matching on
learned prototype parameters, using fold 0 as the reference. This
corrects for k-means permutation ambiguity and gradient drift so that
prototype indices are semantically consistent in downstream analysis.

Expected directory structure from evaluate.py:
    <experiment_dir>/
        best_fold_0.pt
        best_fold_1.pt
        ...
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
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# =====================================================================
# Public API
# =====================================================================

def pool_cv_results(
    eval_dir: str,
    risk_stratification: str = 'median',
    align_prototypes: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Pool predictions and attention data across all CV folds.

    Args:
        eval_dir: Path to evaluation directory containing per-fold outputs.
        risk_stratification: How to split into risk groups.
            'median': median split (2 groups)
            'quartile': quartile split (4 groups)
        align_prototypes: If True, align prototype indices across folds
            using Hungarian matching on trained prototype parameters.
            Requires checkpoint files to be accessible.
        checkpoint_dir: Path to directory containing best_fold_*.pt files.
            If None, inferred as eval_dir parent directory.

    Returns:
        predictions: DataFrame with columns [patient_id, risk_score,
                     survival_time, event, survival_bin, fold, risk_group]
        attention_by_patient: Dict mapping patient_id -> {
            'gene_pathway_attention': ndarray [G, P],
            'pathway_importance': ndarray [P],
            'patch_assignments': dict with soft/hard assignments and gate_weights,
            'cross_modal_attention': ndarray [N, P],
            'fusion_gate_weights': ndarray [N],
            'fold': int
        }
        metadata: Dict with 'gene_names', 'pathway_names',
                  'num_prototypes', 'prototype_alignment', etc.
    """
    eval_dir = Path(eval_dir)

    # --- Pool predictions ---
    predictions = _pool_predictions(eval_dir)

    # --- Assign risk groups ---
    predictions = _assign_risk_groups(predictions, risk_stratification)

    # --- Pool attention data ---
    attention_by_patient, metadata = _pool_attention(eval_dir)

    # --- Align prototypes across folds ---
    if align_prototypes and HAS_TORCH:
        if checkpoint_dir is None:
            checkpoint_dir = eval_dir.parent

        alignment = _align_and_remap_prototypes(
            attention_by_patient, checkpoint_dir
        )
        if alignment is not None:
            metadata['prototype_alignment'] = alignment

    # --- Cross-reference: ensure attention patients match predictions ---
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


# =====================================================================
# Prototype alignment across folds
# =====================================================================

def _align_and_remap_prototypes(
    attention_by_patient: Dict[str, Dict],
    checkpoint_dir,
) -> Optional[Dict]:
    """
    Load trained prototypes from checkpoints, compute alignment to fold 0
    via Hungarian matching, and remap all prototype-indexed signals.

    Args:
        attention_by_patient: Already-pooled attention dict (modified in-place).
        checkpoint_dir: Directory containing best_fold_*.pt checkpoints.

    Returns:
        Alignment diagnostics dict, or None if alignment was skipped.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # --- Load trained prototypes from checkpoints ---
    prototypes = _load_trained_prototypes(checkpoint_dir)
    if prototypes is None:
        return None

    num_folds = len(prototypes)
    if num_folds < 2:
        logger.info("Single fold, no prototype alignment needed")
        return None

    num_prototypes = prototypes[0].shape[0]

    # --- Compute permutations via Hungarian matching ---
    permutations, similarities = _compute_prototype_alignment(prototypes)

    # --- Log diagnostics ---
    _log_alignment_diagnostics(permutations, similarities, num_prototypes)

    # --- Remap attention data ---
    # Group patients by fold
    patients_by_fold = {}
    for pid, attn in attention_by_patient.items():
        fold = attn.get('fold')
        if fold is not None:
            patients_by_fold.setdefault(fold, []).append(pid)

    n_remapped = 0
    for fold_idx, patient_ids in patients_by_fold.items():
        if fold_idx not in permutations:
            continue
        perm = permutations[fold_idx]

        # Skip fold 0 (identity mapping)
        if all(perm[k] == k for k in perm):
            continue

        for pid in patient_ids:
            _remap_patient_attention(
                attention_by_patient[pid], perm, num_prototypes
            )
            n_remapped += 1

    logger.info(
        f"Remapped prototype indices for {n_remapped} patients "
        f"across {num_folds - 1} non-reference folds"
    )

    # --- Build diagnostics output ---
    alignment_info = {
        'reference_fold': 0,
        'num_folds': num_folds,
        'num_prototypes': num_prototypes,
        'permutations': permutations,
        'per_fold_similarities': {
            fold_idx: sims.tolist()
            for fold_idx, sims in similarities.items()
        },
        'per_fold_mean_similarity': {
            fold_idx: float(sims.mean())
            for fold_idx, sims in similarities.items()
        },
        'per_prototype_stability': _compute_per_prototype_stability(
            similarities, num_prototypes, permutations
        ),
    }

    return alignment_info


def _load_trained_prototypes(checkpoint_dir: Path) -> Optional[Dict[int, torch.Tensor]]:
    """
    Load learned prototype parameters from fold checkpoints.

    Tries naming conventions from both train.py and evaluate.py:
        best_fold_0.pt, best_model_fold_0.pt

    Returns:
        Dict[fold_idx -> Tensor [K, D]], or None if no checkpoints found.
    """
    # Try common naming patterns
    patterns = ['best_fold_*.pt', 'best_model_fold_*.pt']
    ckpt_paths = []
    for pattern in patterns:
        ckpt_paths = sorted(checkpoint_dir.glob(pattern))
        if ckpt_paths:
            break

    if not ckpt_paths:
        logger.warning(
            f"No fold checkpoints found in {checkpoint_dir}, "
            f"skipping prototype alignment"
        )
        return None

    prototypes = {}
    proto_key = None

    for ckpt_path in ckpt_paths:
        # Parse fold index
        stem = ckpt_path.stem
        fold_idx = int(stem.split('_')[-1])

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Find the prototype parameter key
        if proto_key is None:
            candidates = [
                'wsi_encoder.proto',
                'wsi_encoder.prototypes',
                'wsi_encoder.proto.weight',
            ]
            for key in candidates:
                if key in state_dict:
                    proto_key = key
                    break

            if proto_key is None:
                # Search for anything with 'proto' in the key
                for key in state_dict:
                    if 'proto' in key.lower() and state_dict[key].ndim == 2:
                        proto_key = key
                        break

            if proto_key is None:
                logger.warning(
                    "Could not find prototype parameters in checkpoint, "
                    "skipping alignment"
                )
                return None

        proto = state_dict[proto_key].detach().clone()
        prototypes[fold_idx] = proto

    logger.info(
        f"Loaded trained prototypes from {len(prototypes)} folds "
        f"(key: '{proto_key}', shape: {proto.shape})"
    )

    return prototypes


def _compute_prototype_alignment(
    prototypes: Dict[int, torch.Tensor],
    reference_fold: int = 0,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, np.ndarray]]:
    """
    Compute optimal prototype permutations aligning each fold to the
    reference fold using Hungarian matching on cosine similarity.

    Args:
        prototypes: Dict[fold_idx -> Tensor [K, D]].
        reference_fold: Fold index to use as reference (default: 0).

    Returns:
        permutations: Dict[fold_idx -> Dict[old_proto_idx -> new_proto_idx]]
            For the reference fold, this is the identity mapping.
        similarities: Dict[fold_idx -> ndarray [K]] of matched cosine sims.
            Not present for the reference fold.
    """
    from scipy.optimize import linear_sum_assignment

    ref = F.normalize(prototypes[reference_fold].float(), dim=1)
    num_prototypes = ref.shape[0]

    permutations = {
        reference_fold: {i: i for i in range(num_prototypes)}
    }
    similarities = {}

    for fold_idx, proto in prototypes.items():
        if fold_idx == reference_fold:
            continue

        proto_norm = F.normalize(proto.float(), dim=1)

        # Cosine similarity matrix [K, K]
        sim = (proto_norm @ ref.T).numpy()

        # Hungarian matching: maximise total similarity
        row_ind, col_ind = linear_sum_assignment(-sim)

        # row_ind[i] in this fold -> col_ind[i] in reference fold
        permutations[fold_idx] = dict(
            zip(row_ind.tolist(), col_ind.tolist())
        )
        similarities[fold_idx] = sim[row_ind, col_ind]

    return permutations, similarities


def _remap_patient_attention(
    patient_attn: Dict,
    permutation: Dict[int, int],
    num_prototypes: int,
):
    """
    Remap all prototype-indexed signals for a single patient (in-place).

    Handles:
        - cross_modal_attention [K, P] -> reorder rows
        - patch_assignments.gate_weights [K] -> reorder
        - fusion_gate_weights [K] -> reorder
        - patch_assignments.assignments [N_patches] -> relabel values
        - patch_assignments.soft_assignments [N_patches, K] -> reorder cols
    """
    # Build reorder index: new_position[i] should come from old_position
    # permutation says old->new, we need inv: new->old
    inv_perm = {v: k for k, v in permutation.items()}
    reorder = np.array([inv_perm[i] for i in range(num_prototypes)])

    # Signal G: cross-modal attention [K, P]
    if 'cross_modal_attention' in patient_attn:
        cm = patient_attn['cross_modal_attention']
        if hasattr(cm, '__len__') and len(cm) == num_prototypes:
            patient_attn['cross_modal_attention'] = cm[reorder]

    # Signal E: WSI gate weights [K]
    if isinstance(patient_attn.get('patch_assignments'), dict):
        pa = patient_attn['patch_assignments']

        if 'gate_weights' in pa:
            gw = pa['gate_weights']
            if hasattr(gw, '__len__') and len(gw) == num_prototypes:
                pa['gate_weights'] = gw[reorder]

        # Hard assignments: relabel prototype IDs per patch
        assign_key = (
            'hard_assignments' if 'hard_assignments' in pa
            else 'assignments' if 'assignments' in pa
            else None
        )
        if assign_key is not None:
            old_assignments = np.asarray(pa[assign_key])
            pa[assign_key] = np.array(
                [permutation.get(int(a), int(a)) for a in old_assignments]
            )

        # Soft assignments [N_patches, K]: reorder columns
        if 'soft_assignments' in pa:
            soft = pa['soft_assignments']
            if hasattr(soft, 'shape') and soft.ndim == 2 and soft.shape[1] == num_prototypes:
                pa['soft_assignments'] = soft[:, reorder]

    # Signal H: fusion gate weights [K]
    if 'fusion_gate_weights' in patient_attn:
        fg = patient_attn['fusion_gate_weights']
        if hasattr(fg, '__len__') and len(fg) == num_prototypes:
            patient_attn['fusion_gate_weights'] = fg[reorder]


def _compute_per_prototype_stability(
    similarities: Dict[int, np.ndarray],
    num_prototypes: int,
    permutations: Optional[Dict[int, Dict[int, int]]] = None,
) -> List[Dict]:
    """
    Compute per-prototype stability across folds.

    For each prototype (in reference fold numbering), reports the mean
    and min cosine similarity with its matched counterpart across folds.

    Args:
        similarities: Dict[fold_idx -> ndarray [K]] of matched cosine sims.
        num_prototypes: Total number of prototypes.
        permutations: Dict[fold_idx -> Dict[old -> new (ref)]].
            Needed to map sim values back to reference prototype indices.

    Returns:
        List of dicts with per-prototype stability metrics.
    """
    if not similarities:
        return []

    # Gather sims indexed by reference prototype
    proto_sims = {i: [] for i in range(num_prototypes)}

    for fold_idx, sims in similarities.items():
        if permutations is not None and fold_idx in permutations:
            perm = permutations[fold_idx]
            # perm maps old_idx -> ref_idx
            # sims[j] is the cosine sim for this fold's proto j
            # which maps to reference proto perm[j]
            for old_idx in range(num_prototypes):
                ref_idx = perm[old_idx]
                proto_sims[ref_idx].append(sims[old_idx])
        else:
            # Fall back: assume ordering matches
            for j in range(num_prototypes):
                proto_sims[j].append(sims[j])

    stability = []
    for ref_idx in range(num_prototypes):
        fold_sims = proto_sims[ref_idx]
        if fold_sims:
            arr = np.array(fold_sims)
            stability.append({
                'prototype_idx': ref_idx,
                'mean_cosine_sim': float(arr.mean()),
                'min_cosine_sim': float(arr.min()),
                'std_cosine_sim': float(arr.std()),
                'n_folds': len(arr),
                'stable': bool(arr.min() >= 0.7),
            })
        else:
            stability.append({
                'prototype_idx': ref_idx,
                'mean_cosine_sim': None,
                'min_cosine_sim': None,
                'std_cosine_sim': None,
                'n_folds': 0,
                'stable': False,
            })

    return stability


def _log_alignment_diagnostics(
    permutations: Dict[int, Dict[int, int]],
    similarities: Dict[int, np.ndarray],
    num_prototypes: int,
):
    """Log a summary of prototype alignment quality."""
    logger.info("=" * 50)
    logger.info("Prototype Alignment Diagnostics (ref: fold 0)")
    logger.info("=" * 50)

    for fold_idx in sorted(similarities.keys()):
        sims = similarities[fold_idx]
        perm = permutations[fold_idx]

        # Check if permutation is identity
        is_identity = all(perm[k] == k for k in sorted(perm.keys()))

        # Check how many prototypes are permuted
        n_permuted = sum(1 for k, v in perm.items() if k != v)

        logger.info(
            f"  Fold {fold_idx}: "
            f"mean cos sim = {sims.mean():.4f}, "
            f"min = {sims.min():.4f}, "
            f"max = {sims.max():.4f}, "
            f"permuted = {n_permuted}/{num_prototypes}"
            f"{' (identity)' if is_identity else ''}"
        )

        # Flag any poorly matched prototypes
        weak_mask = sims < 0.7
        if weak_mask.any():
            weak_indices = np.where(weak_mask)[0]
            logger.warning(
                f"    ⚠ Weak matches (cos < 0.7): "
                f"prototypes {weak_indices.tolist()} "
                f"with sims {sims[weak_indices].tolist()}"
            )

    # Overall summary
    all_sims = np.concatenate(list(similarities.values()))
    logger.info(
        f"  Overall: {len(all_sims)} matches, "
        f"mean = {all_sims.mean():.4f}, "
        f"min = {all_sims.min():.4f}, "
        f"std = {all_sims.std():.4f}"
    )

    if all_sims.min() < 0.5:
        logger.warning(
            "  ⚠ Some prototypes have very low cross-fold similarity. "
            "Consider these unstable in downstream analysis."
        )


def save_alignment_report(
    alignment_info: Dict,
    output_path: str,
):
    """
    Save a detailed alignment report to disk.

    Args:
        alignment_info: Dict from metadata['prototype_alignment'].
        output_path: Path to save (text or JSON).
    """
    if alignment_info is None:
        return

    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    lines = []
    lines.append("Prototype Alignment Report")
    lines.append("=" * 60)
    lines.append(f"Reference fold: {alignment_info['reference_fold']}")
    lines.append(f"Number of folds: {alignment_info['num_folds']}")
    lines.append(f"Number of prototypes: {alignment_info['num_prototypes']}")
    lines.append("")

    lines.append("Per-fold alignment:")
    lines.append("-" * 40)
    for fold_idx, mean_sim in sorted(
        alignment_info['per_fold_mean_similarity'].items()
    ):
        sims = alignment_info['per_fold_similarities'][fold_idx]
        perm = alignment_info['permutations'][fold_idx]
        n_permuted = sum(1 for k, v in perm.items() if k != v)
        lines.append(
            f"  Fold {fold_idx}: mean_sim={mean_sim:.4f}, "
            f"min_sim={min(sims):.4f}, "
            f"permuted={n_permuted}/{alignment_info['num_prototypes']}"
        )

    lines.append("")
    lines.append("Permutation mappings (old -> new):")
    lines.append("-" * 40)
    for fold_idx, perm in sorted(alignment_info['permutations'].items()):
        if fold_idx == alignment_info['reference_fold']:
            continue
        non_identity = {k: v for k, v in perm.items() if k != v}
        if non_identity:
            lines.append(f"  Fold {fold_idx}: {non_identity}")
        else:
            lines.append(f"  Fold {fold_idx}: (identity)")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Saved alignment report to {output_path}")


# =====================================================================
# Predictions and attention pooling (unchanged from original)
# =====================================================================

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
    right-inclusive behaviour: (q1, q2] -> group 1.

    Args:
        risk_scores: 1-D array of (normalised) risk scores.
        n_groups: 2 (median split) or 4 (quartile split).

    Returns:
        groups: int array of group indices (0 ... n_groups-1).
        group_names: list of display labels.
    """
    if n_groups == 2:
        threshold = np.median(risk_scores)
        groups = (risk_scores > threshold).astype(int)
        return groups, GROUP_NAMES_2
    elif n_groups == 4:
        quartiles = np.percentile(risk_scores, [25, 50, 75])
        # searchsorted(side='right'): value == boundary -> stays in lower bin
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
    experiment config.
    """
    names = {}

    # Try to load from attention pickle metadata first
    eval_dir = Path(eval_dir)
    attn_dirs = sorted(eval_dir.glob('attention_fold_*'))
    if not attn_dirs:
        attn_dirs = sorted(eval_dir.glob('attention*'))

    for attn_dir in attn_dirs:
        pkl_path = Path(attn_dir) / 'attention_weights.pkl'
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if data.get('gene_names'):
                names['gene_names'] = data['gene_names']
            if data.get('pathway_names'):
                names['pathway_names'] = data['pathway_names']
            if names:
                break

    return names