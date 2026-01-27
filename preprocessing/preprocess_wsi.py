"""
WSI Feature Preprocessing

Loads pre-extracted WSI features (e.g., UNI2-h from Mahmood lab HuggingFace)
and organizes them into a format suitable for training.

Pipeline:
1. Scan directory for H5 feature files
2. Extract patient IDs from filenames
3. Load features and organize by patient
4. Handle multiple slides per patient (concatenate patches)
5. Filter to patients with labels
6. Save as pickle dict

Usage:
    python main.py preprocess wsi --config configs/preprocessing/preprocess_wsi.yaml
    python main.py preprocess wsi --config configs/preprocessing/preprocess_wsi.yaml dataset=HNSC
"""

import os
import json
import logging
import pickle
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def find_h5_files(directory):
    """Find all H5 files in a directory."""
    h5_files = []
    for ext in ['.h5', '.hdf5']:
        h5_files.extend(Path(directory).glob(f'*{ext}'))
    return sorted(h5_files)


def extract_patient_id(filename, expression):
    """
    Extract patient ID from filename using a Python expression.

    Args:
        filename: The filename (without path)
        expression: Python expression to evaluate with 'filename' as variable

    Returns:
        Extracted patient ID
    """
    # Make filename available to the expression
    return eval(expression)


def load_h5_features(h5_path):
    """
    Load features from an H5 file.

    Args:
        h5_path: Path to H5 file

    Returns:
        numpy array of shape [num_patches, feature_dim]
    """
    with h5py.File(h5_path, 'r') as f:
        # Try common key names
        if 'features' in f:
            features = f['features'][:]
        elif 'feats' in f:
            features = f['feats'][:]
        else:
            # Use first available dataset
            key = list(f.keys())[0]
            features = f[key][:]

        # Squeeze if needed (some extractors add extra dimensions)
        features = features.squeeze()

        # Ensure 2D: [num_patches, feature_dim]
        if features.ndim == 1:
            features = features.reshape(1, -1)

        return features


def plot_feature_statistics(patient_features, output_path):
    """Plot statistics about the loaded features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Patches per patient
    patches_per_patient = [f.shape[0] for f in patient_features.values()]
    axes[0].hist(patches_per_patient, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Patches per Patient')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Patches per Patient (median: {np.median(patches_per_patient):.0f})')
    axes[0].axvline(np.median(patches_per_patient), color='red', linestyle='--')

    # Feature dimension check (should be constant)
    dims = [f.shape[1] for f in patient_features.values()]
    unique_dims = set(dims)
    axes[1].bar(range(len(unique_dims)), [dims.count(d) for d in unique_dims])
    axes[1].set_xticks(range(len(unique_dims)))
    axes[1].set_xticklabels(list(unique_dims))
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Feature Dimensions')

    # Sample feature distribution (first patient)
    sample_patient = list(patient_features.keys())[0]
    sample_features = patient_features[sample_patient].flatten()
    axes[2].hist(sample_features, bins=100, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Feature Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Feature Distribution (sample: {sample_patient})')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved feature statistics plot to {output_path}")


def plot_slides_per_patient(slides_per_patient, output_path):
    """Plot distribution of slides per patient."""
    counts = list(slides_per_patient.values())

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(1, max(counts) + 2), edgecolor='black', alpha=0.7, align='left')
    plt.xlabel('Number of Slides per Patient')
    plt.ylabel('Number of Patients')
    plt.title(f'Slides per Patient Distribution (n={len(counts)})')
    plt.xticks(range(1, max(counts) + 1))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logging.info(f"Saved slides per patient plot to {output_path}")


def run(cfg):
    """
    Run the WSI preprocessing pipeline.

    Args:
        cfg: OmegaConf configuration object

    Returns:
        dict: Paths to output files
    """
    dataset = cfg['dataset']

    logging.info("=" * 60)
    logging.info(f"WSI Feature Preprocessing: {dataset}")
    logging.info("=" * 60)

    # Create output directories
    output_dir = cfg['output']['dir']
    figures_dir = cfg['output']['figures_dir']
    generate_figures = cfg['output'].get('generate_figures', True)

    os.makedirs(output_dir, exist_ok=True)
    if generate_figures:
        os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Find H5 files
    # -------------------------------------------------------------------------
    logging.info("\n[Step 1/5] Scanning for H5 feature files...")

    features_dir = cfg['input']['features_dir']
    h5_files = find_h5_files(features_dir)

    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {features_dir}")

    logging.info(f"Found {len(h5_files)} H5 files")

    # -------------------------------------------------------------------------
    # Step 2: Extract patient IDs and organize by patient
    # -------------------------------------------------------------------------
    logging.info("\n[Step 2/5] Extracting patient IDs from filenames...")

    patient_id_expr = cfg['parsing']['patient_id_expression']
    slide_type_filter = cfg['parsing'].get('slide_type_filter')

    # Group files by patient
    patient_files = defaultdict(list)
    skipped_files = 0

    for h5_path in h5_files:
        filename = h5_path.name

        # Apply slide type filter if specified
        if slide_type_filter and slide_type_filter not in filename:
            skipped_files += 1
            continue

        try:
            patient_id = extract_patient_id(filename, patient_id_expr)
            patient_files[patient_id].append(h5_path)
        except Exception as e:
            logging.warning(f"Could not parse patient ID from {filename}: {e}")
            skipped_files += 1

    logging.info(f"Found {len(patient_files)} unique patients")
    if skipped_files > 0:
        logging.info(f"Skipped {skipped_files} files (filter or parse error)")

    # Count slides per patient
    slides_per_patient = {pid: len(files) for pid, files in patient_files.items()}
    multi_slide_patients = sum(1 for c in slides_per_patient.values() if c > 1)
    logging.info(f"Patients with multiple slides: {multi_slide_patients}")

    if generate_figures and multi_slide_patients > 0:
        plot_slides_per_patient(
            slides_per_patient,
            os.path.join(figures_dir, 'slides_per_patient.pdf')
        )

    # -------------------------------------------------------------------------
    # Step 3: Load features
    # -------------------------------------------------------------------------
    logging.info("\n[Step 3/5] Loading features from H5 files...")

    max_slides = cfg.get('max_slides_per_patient', 2)

    patient_features = {}
    feature_dim = None
    load_errors = 0
    slides_truncated = 0

    for i, (patient_id, files) in enumerate(patient_files.items()):
        if (i + 1) % 50 == 0:
            logging.info(f"  Loaded {i + 1}/{len(patient_files)} patients...")

        try:
            # Load all slides for this patient
            slide_features = []
            for h5_path in files:
                features = load_h5_features(h5_path)
                slide_features.append((h5_path, features))

            # If more than max_slides, keep only the largest ones (by patch count)
            if len(slide_features) > max_slides:
                slides_truncated += 1
                # Sort by number of patches (descending) and keep top max_slides
                slide_features.sort(key=lambda x: x[1].shape[0], reverse=True)
                slide_features = slide_features[:max_slides]
                logging.debug(f"Patient {patient_id}: truncated to {max_slides} largest slides")

            # Concatenate patches from selected slides
            all_patches = [sf[1] for sf in slide_features]
            if len(all_patches) > 1:
                combined = np.concatenate(all_patches, axis=0)
            else:
                combined = all_patches[0]

            # Verify feature dimension
            if feature_dim is None:
                feature_dim = combined.shape[1]
            elif combined.shape[1] != feature_dim:
                logging.warning(f"Feature dimension mismatch for {patient_id}: "
                                f"expected {feature_dim}, got {combined.shape[1]}")
                load_errors += 1
                continue

            patient_features[patient_id] = combined

        except Exception as e:
            logging.warning(f"Error loading features for {patient_id}: {e}")
            load_errors += 1

    logging.info(f"Successfully loaded features for {len(patient_features)} patients")
    if load_errors > 0:
        logging.warning(f"Failed to load {load_errors} patients")
    if slides_truncated > 0:
        logging.info(f"Truncated to {max_slides} slides for {slides_truncated} patients")

    # -------------------------------------------------------------------------
    # Step 4: Filter to patients with labels (optional)
    # -------------------------------------------------------------------------
    logging.info("\n[Step 4/5] Filtering to patients with labels...")

    labels_path = cfg['input'].get('labels')

    if labels_path and os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        patient_id_col = cfg['columns']['patient_id']

        labeled_patients = set(labels_df[patient_id_col].astype(str))

        before_filter = len(patient_features)
        patient_features = {
            pid: feats for pid, feats in patient_features.items()
            if pid in labeled_patients
        }

        logging.info(f"Filtered to patients with labels: {before_filter} → {len(patient_features)}")
    else:
        logging.info("No labels file provided, keeping all patients")

    # -------------------------------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------------------------------
    logging.info("\n[Step 5/5] Saving outputs...")

    # Convert to tensors
    patient_features_tensor = {
        pid: torch.tensor(feats, dtype=torch.float32)
        for pid, feats in patient_features.items()
    }

    # Save as pickle
    features_path = os.path.join(output_dir, 'wsi_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(patient_features_tensor, f)
    logging.info(f"Saved WSI features to {features_path}")

    # Plot feature statistics
    if generate_figures:
        plot_feature_statistics(
            patient_features,
            os.path.join(figures_dir, 'wsi_feature_statistics.pdf')
        )

    # Compute and save statistics
    total_patches = sum(f.shape[0] for f in patient_features.values())
    patches_per_patient = [f.shape[0] for f in patient_features.values()]

    info = {
        'dataset': dataset,
        'num_patients': len(patient_features),
        'feature_dim': feature_dim,
        'total_patches': total_patches,
        'mean_patches_per_patient': np.mean(patches_per_patient),
        'median_patches_per_patient': np.median(patches_per_patient),
        'min_patches': min(patches_per_patient),
        'max_patches': max(patches_per_patient),
        'feature_extractor': cfg['feature_extractor']['name'],
        'slide_type_filter': slide_type_filter,
        'patient_ids': list(patient_features.keys())
    }

    info_path = os.path.join(output_dir, 'wsi_preprocessing_info.json')
    with open(info_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else
                       (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                   for k, v in info.items()}, f, indent=2)
    logging.info(f"Saved preprocessing info to {info_path}")

    # Save config snapshot
    from omegaconf import OmegaConf
    config_path = os.path.join(output_dir, 'preprocess_wsi_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logging.info(f"Saved config to {config_path}")

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("WSI Preprocessing Summary")
    logging.info("=" * 60)
    logging.info(f"  Dataset: {dataset}")
    logging.info(f"  Patients: {len(patient_features)}")
    logging.info(f"  Feature dimension: {feature_dim}")
    logging.info(f"  Total patches: {total_patches}")
    logging.info(f"  Mean patches/patient: {np.mean(patches_per_patient):.1f}")
    logging.info(f"  Median patches/patient: {np.median(patches_per_patient):.0f}")
    logging.info(f"  Feature extractor: {cfg['feature_extractor']['name']}")

    logging.info("\n" + "=" * 60)
    logging.info("WSI preprocessing complete!")
    logging.info("=" * 60)

    return {
        'features': features_path,
        'info': info_path
    }