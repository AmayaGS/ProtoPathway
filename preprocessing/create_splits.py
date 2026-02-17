"""
Create Splits and Labels (Stage 3)

Finds shared patients between modalities and creates train/val/test splits.
Handles both predefined TCGA splits (survival) and generated splits (classification).

Pipeline:
1. Load patient IDs from gene expression and WSI preprocessing outputs
2. Find intersection (multimodal patients)
3. Load clinical data and process labels
4. Either load predefined TCGA splits OR generate stratified k-fold splits
5. Filter splits to available patients
6. Save splits, labels, and shared patient IDs

Usage:
    python main.py preprocess splits --config configs/preprocessing/create_splits.yaml
    python main.py preprocess splits --config configs/preprocessing/create_splits.yaml dataset=HNSC
"""

import os
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.survival import (
    discretize_survival_times,
    compute_survival_statistics
)


def load_patient_ids_from_expression(expression_path):
    """Load patient IDs from gene expression CSV (index column)."""
    df = pd.read_csv(expression_path, index_col=0)
    patient_ids = set(str(pid) for pid in df.index)
    logging.info(f"Loaded {len(patient_ids)} patient IDs from gene expression")
    return patient_ids


def load_patient_ids_from_wsi(patient_ids_csv_path):
    """
    Load patient IDs from lightweight CSV file (output of preprocess_wsi).

    This avoids loading the full WSI features pickle which can be very large.
    """
    df = pd.read_csv(patient_ids_csv_path)
    patient_ids = set(str(pid) for pid in df['patient_id'])
    logging.info(f"Loaded {len(patient_ids)} patient IDs from WSI")
    return patient_ids


def load_predefined_tcga_splits(splits_dir, available_patients):
    """
    Load predefined TCGA splits from CSV files.

    Expected format (from SurvPath repository):
        splits_dir/
            splits_0.csv  (columns: train, val)  # No test column - pure CV
            splits_1.csv
            ...

    This is pure k-fold CV where each fold's validation set rotates.
    There is no held-out test set - all patients appear in validation exactly once.

    Args:
        splits_dir: Directory containing split CSV files
        available_patients: Set of patient IDs we have data for

    Returns:
        dict: Standard split dictionary format (Test will be empty for pure CV)
    """
    splits_dir = Path(splits_dir)
    split_files = sorted(splits_dir.glob('splits_*.csv'))

    if not split_files:
        raise FileNotFoundError(f"No split files found in {splits_dir}")

    logging.info(f"Found {len(split_files)} predefined split files")

    # Check first file to see if test column exists
    sample_df = pd.read_csv(split_files[0])
    has_test_column = 'test' in sample_df.columns
    logging.info(f"  Split format: {'train/val/test' if has_test_column else 'train/val only (pure CV)'}")

    split_dict = {
        'Train': [],  # All patients used in training across folds
        'Test': [],   # Empty for pure CV, populated if test column exists
        'CV': {}
    }

    all_patients = set()
    test_patients = None

    for i, split_file in enumerate(split_files):
        df = pd.read_csv(split_file)

        # Extract patient IDs from each column
        train_ids = df['train'].dropna().astype(str).tolist()
        val_ids = df['val'].dropna().astype(str).tolist()

        # Filter to available patients
        train_filtered = [pid for pid in train_ids if pid in available_patients]
        val_filtered = [pid for pid in val_ids if pid in available_patients]

        # Handle optional test column
        if has_test_column:
            test_ids = df['test'].dropna().astype(str).tolist()
            test_filtered = [pid for pid in test_ids if pid in available_patients]
            logging.info(f"  Fold {i}: train {len(train_ids)}→{len(train_filtered)}, "
                         f"val {len(val_ids)}→{len(val_filtered)}, "
                         f"test {len(test_ids)}→{len(test_filtered)}")

            if test_patients is None:
                test_patients = set(test_filtered)
            elif set(test_filtered) != test_patients:
                logging.warning(f"Test set differs in fold {i}")
        else:
            logging.info(f"  Fold {i}: train {len(train_ids)}→{len(train_filtered)}, "
                         f"val {len(val_ids)}→{len(val_filtered)}")

        # Store CV fold
        fold_name = f"Fold {i}"
        split_dict['CV'][fold_name] = {
            'Train': train_filtered,
            'Val': val_filtered
        }

        all_patients.update(train_filtered)
        all_patients.update(val_filtered)

    # For pure CV, Train contains all patients (they all appear in some fold's training)
    # For train/val/test format, Train excludes test patients
    split_dict['Train'] = list(all_patients)
    split_dict['Test'] = list(test_patients) if test_patients else []

    return split_dict


def generate_stratified_splits(patient_ids, labels, cfg):
    """
    Generate stratified k-fold CV splits.

    By default, creates pure k-fold CV (matching TCGA predefined format).
    Optionally can create a held-out test set if test_fraction > 0.

    Args:
        patient_ids: List of patient IDs
        labels: List of labels (for stratification)
        cfg: Split generation config with test_fraction, num_folds, seed

    Returns:
        dict: Standard split dictionary format
    """
    test_fraction = cfg.get('test_fraction', 0.0)  # Default: no held-out test set
    num_folds = cfg.get('num_folds', 5)
    seed = cfg.get('seed', 42)

    patient_ids = np.array(patient_ids)
    labels = np.array(labels)

    # Optionally split off a held-out test set
    if test_fraction > 0:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(patient_ids)),
            test_size=test_fraction,
            stratify=labels,
            random_state=seed
        )
        train_val_patients = patient_ids[train_val_idx]
        train_val_labels = labels[train_val_idx]
        test_patients = patient_ids[test_idx].tolist()
        logging.info(f"Split off held-out test set: {len(test_patients)} patients ({test_fraction:.0%})")
    else:
        # Pure CV - no held-out test set
        train_val_patients = patient_ids
        train_val_labels = labels
        test_patients = []
        logging.info("Pure CV mode (no held-out test set)")

    logging.info(f"Creating {num_folds}-fold CV on {len(train_val_patients)} patients")

    # K-fold CV
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    split_dict = {
        'Train': train_val_patients.tolist(),  # All patients in CV
        'Test': test_patients,
        'CV': {}
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_patients, train_val_labels)):
        fold_name = f"Fold {fold_idx}"
        split_dict['CV'][fold_name] = {
            'Train': train_val_patients[train_idx].tolist(),
            'Val': train_val_patients[val_idx].tolist()
        }
        logging.info(f"  {fold_name}: train {len(train_idx)}, val {len(val_idx)}")

    return split_dict


def validate_splits(split_dict):
    """Validate that splits have no overlaps."""
    test_set = set(split_dict['Test'])

    for fold_name, fold_splits in split_dict['CV'].items():
        train_set = set(fold_splits['Train'])
        val_set = set(fold_splits['Val'])

        # Check overlaps
        assert train_set.isdisjoint(val_set), f"{fold_name}: Train/Val overlap"
        assert train_set.isdisjoint(test_set), f"{fold_name}: Train/Test overlap"
        assert val_set.isdisjoint(test_set), f"{fold_name}: Val/Test overlap"

    logging.info("Split validation passed: no overlaps detected")


def run(cfg):
    """
    Run the split creation pipeline.

    Args:
        cfg: OmegaConf configuration object

    Returns:
        dict: Paths to output files
    """
    dataset_name = cfg.get('dataset_name', cfg.get('dataset', 'unknown'))

    logging.info("=" * 60)
    logging.info(f"Create Splits and Labels: {dataset_name}")
    logging.info("=" * 60)

    # Create output directory
    output_dir = cfg['paths']['processed_dir']
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load patient IDs from both modalities
    # -------------------------------------------------------------------------
    logging.info("\n[Step 1/5] Loading patient IDs from preprocessed data...")

    gene_patient_ids = load_patient_ids_from_expression(cfg['input']['gene_expression'])
    wsi_patient_ids = load_patient_ids_from_wsi(cfg['input']['wsi_patient_ids'])

    # Find intersection
    shared_patient_ids = gene_patient_ids & wsi_patient_ids
    logging.info(f"Shared patients (multimodal): {len(shared_patient_ids)}")

    if len(shared_patient_ids) == 0:
        raise ValueError("No shared patients between gene expression and WSI data!")

    # -------------------------------------------------------------------------
    # Step 2: Load clinical data
    # -------------------------------------------------------------------------
    logging.info("\n[Step 2/5] Loading clinical data...")

    clinical_df = pd.read_csv(cfg['input']['clinical_csv'])
    patient_id_col = cfg['patient_id_col']

    logging.info(f"Loaded clinical CSV: {len(clinical_df)} rows")

    # Ensure patient ID column is string
    clinical_df[patient_id_col] = clinical_df[patient_id_col].astype(str)

    # Drop duplicate patient entries (keep first occurrence)
    n_before = len(clinical_df)
    clinical_df = clinical_df.drop_duplicates(subset=[patient_id_col], keep='first')
    if len(clinical_df) < n_before:
        logging.info(f"Removed {n_before - len(clinical_df)} duplicate rows, {len(clinical_df)} unique patients")

    # Filter to shared patients (intersection of gene, WSI, and clinical)
    clinical_df = clinical_df[clinical_df[patient_id_col].isin(shared_patient_ids)].copy()

    # Update shared_patient_ids to only those with clinical data
    shared_patient_ids = set(clinical_df[patient_id_col])
    logging.info(f"Patients with all modalities + clinical data: {len(shared_patient_ids)}")

    # -------------------------------------------------------------------------
    # Step 3: Process labels based on task type
    # -------------------------------------------------------------------------
    logging.info("\n[Step 3/5] Processing labels...")

    task = cfg['task']

    if task == 'survival':
        time_col = cfg['survival']['time_col']
        censorship_col = cfg['survival']['event_col']  # Raw TCGA censorship column
        num_bins = cfg['survival']['num_bins']

        # Convert censorship (1=alive) to event (1=death) for all downstream use
        clinical_df['event'] = 1 - clinical_df[censorship_col]

        # Discretize survival times (adds 'survival_bin' column to clinical_df)
        clinical_df, bin_edges = discretize_survival_times(
            clinical_df, time_col, 'event', n_bins=num_bins
        )

        # Build label dictionary directly
        label_dict = {}
        for _, row in clinical_df.iterrows():
            pid = str(row[patient_id_col])
            label_dict[pid] = {
                'time': float(row[time_col]),
                'event': int(row['event']),
                'bin': int(row['survival_bin'])
            }

        # For stratification, use survival bins
        stratification_labels = [label_dict[pid]['bin'] for pid in shared_patient_ids]

        # Compute and log survival statistics
        times = clinical_df[time_col].values
        events = clinical_df['event'].values
        stats = compute_survival_statistics(times, events)
        logging.info(f"Survival statistics:")
        logging.info(f"  Total patients: {stats['n_total']}")
        logging.info(f"  Events: {stats['n_events']} ({stats['event_rate']:.1%})")
        logging.info(f"  Censored: {stats['n_censored']}")
        if stats['median_time_events'] is not None:
            logging.info(f"  Median survival (events): {stats['median_time_events']:.1f}")

    elif task == 'classification':
        label_col = cfg['classification']['label_col']

        # Build label dictionary
        label_dict = {}
        for _, row in clinical_df.iterrows():
            pid = str(row[patient_id_col])
            label_dict[pid] = int(row[label_col])

        # For stratification
        stratification_labels = [label_dict[pid] for pid in shared_patient_ids]

        # Log class distribution
        label_counts = pd.Series(stratification_labels).value_counts()
        logging.info("Class distribution:")
        for label, count in label_counts.items():
            logging.info(f"  Class {label}: {count} ({count/len(stratification_labels):.1%})")

        # Placeholder for consistency
        bin_edges = None
        stats = None

    else:
        raise ValueError(f"Unknown task type: {task}")

    # -------------------------------------------------------------------------
    # Step 4: Create or load splits
    # -------------------------------------------------------------------------
    logging.info("\n[Step 4/5] Creating/loading splits...")

    splits_cfg = cfg['splits']
    shared_patient_list = list(shared_patient_ids)

    if splits_cfg['use_predefined']:
        logging.info("Loading predefined TCGA splits...")
        split_dict = load_predefined_tcga_splits(
            splits_cfg['predefined_dir'],
            shared_patient_ids
        )
    else:
        logging.info("Generating stratified splits...")
        split_dict = generate_stratified_splits(
            shared_patient_list,
            stratification_labels,
            splits_cfg['generate']
        )

    # Validate splits
    validate_splits(split_dict)

    # -------------------------------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------------------------------
    logging.info("\n[Step 5/5] Saving outputs...")

    # 1. Save splits
    splits_path = cfg['output']['splits']
    with open(splits_path, 'wb') as f:
        pickle.dump(split_dict, f)
    logging.info(f"Saved splits to {splits_path}")

    # 2. Save labels
    labels_path = cfg['output']['labels']

    if task == 'survival':
        # Save full survival info
        labels_df = pd.DataFrame([
            {
                patient_id_col: pid,
                'survival_time': label_dict[pid]['time'],
                'event': int(label_dict[pid]['event']),
                'survival_bin': label_dict[pid]['bin']
            }
            for pid in shared_patient_ids
        ])
    else:
        labels_df = pd.DataFrame([
            {patient_id_col: pid, 'label': label_dict[pid]}
            for pid in shared_patient_ids
        ])

    labels_df.to_csv(labels_path, index=False)
    logging.info(f"Saved labels to {labels_path}")

    # 3. Save shared patient IDs
    shared_ids_path = cfg['output']['shared_patient_ids']
    pd.DataFrame({patient_id_col: list(shared_patient_ids)}).to_csv(shared_ids_path, index=False)
    logging.info(f"Saved shared patient IDs to {shared_ids_path}")

    # 4. Save preprocessing info
    info = {
        'dataset': dataset_name,
        'task': task,
        'num_gene_patients': len(gene_patient_ids),
        'num_wsi_patients': len(wsi_patient_ids),
        'num_shared_patients': len(shared_patient_ids),
        'num_folds': len(split_dict['CV']),
        'num_train': len(split_dict['Train']),
        'num_test': len(split_dict['Test']),
        'use_predefined_splits': splits_cfg['use_predefined']
    }

    if task == 'survival' and stats is not None:
        info['survival_bins'] = num_bins
        info['survival_stats'] = stats
        if bin_edges is not None:
            info['bin_edges'] = bin_edges.tolist() if hasattr(bin_edges, 'tolist') else list(bin_edges)

    info_path = os.path.join(output_dir, 'splits_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logging.info(f"Saved info to {info_path}")

    # 5. Save config snapshot
    from omegaconf import OmegaConf
    config_path = os.path.join(output_dir, 'create_splits_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.info(f"Saved config to {config_path}")

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Split Creation Summary")
    logging.info("=" * 60)
    logging.info(f"  Dataset: {dataset_name}")
    logging.info(f"  Task: {task}")
    logging.info(f"  Shared patients: {len(shared_patient_ids)}")
    logging.info(f"  Train patients: {len(split_dict['Train'])}")
    logging.info(f"  Test patients: {len(split_dict['Test'])}")
    logging.info(f"  CV folds: {len(split_dict['CV'])}")

    for fold_name, fold_data in split_dict['CV'].items():
        logging.info(f"    {fold_name}: {len(fold_data['Train'])} train, {len(fold_data['Val'])} val")

    logging.info("\n" + "=" * 60)
    logging.info("Split creation complete!")
    logging.info("=" * 60)

    return {
        'splits': splits_path,
        'labels': labels_path,
        'shared_patient_ids': shared_ids_path,
        'info': info_path
    }