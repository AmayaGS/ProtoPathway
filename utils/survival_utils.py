
# utils/survival_utils.py

import os
import numpy as np
import pandas as pd
import pickle

import torch

from utils.helpers import ensure_directory

def load_tcga_splits(config, splits_dir):
    """
    Load TCGA splits from predefined CSV files and create a dictionary with the same
    structure as create_cross_validation_splits for compatibility.

    Args:
        config: Configuration dictionary
        splits_dir: Directory containing the split files (named split_0.csv, split_1.csv, etc.)
        test_ids: Optional list of test patient IDs (if None, will look for test.csv or generate)

    Returns:
        Dictionary with CV, Train and Test split IDs
    """

    # Output path
    output_path = os.path.join(config['output']['data']['dir'], f"data_splits_{config['dataset_name']}.pkl")

    # load metadata to filter down to available patients
    metadata_path = os.path.join(config['input']['dir'], config['input']['patient_labels'])
    metadata_df = pd.read_csv(metadata_path)

    # filtered metadata path
    filtered_metadata_path = config['output']['data']['filtered_labels']

    all_patient_ids = []

    # Initialize dictionary structure
    split_dict = {
        "Train": [],
        "Test": [],
        "CV": {}
    }

    # Get list of split files
    split_files = [f for f in os.listdir(splits_dir) if f.startswith("splits_") and f.endswith(".csv")]
    split_files.sort()  # Sort to ensure consistent ordering

    print(f"Found {len(split_files)} split files in {splits_dir}")

    # First process split_0 for the main Train/Test set
    first_split_file = "splits_0.csv"
    if first_split_file in split_files:
        file_path = os.path.join(splits_dir, first_split_file)
        try:
            # Read CSV with header
            split_df = pd.read_csv(file_path)

            # Extract train and val IDs
            train_ids = split_df['train'].dropna().tolist()
            val_ids = split_df['val'].dropna().tolist()

            # Set main Train set
            split_dict["Train"] = train_ids

            # Use val IDs as Test if no test_ids provided
            if not split_dict["Test"]:
                split_dict["Test"] = val_ids

            print(f"Main split: {len(train_ids)} train samples, {len(val_ids)} validation/test samples")

        except Exception as e:
            print(f"Error reading {first_split_file}: {e}")
    else:
        print(f"Critical error: {first_split_file} not found in {splits_dir}")

    # Now process all splits for the CV section
    for split_file in split_files:
        # Extract fold index from filename
        fold_idx = int(split_file.split("_")[1].split(".")[0])
        file_path = os.path.join(splits_dir, split_file)

        try:
            # Read CSV with header
            split_df = pd.read_csv(file_path)

            # Extract train and val IDs
            train_ids = split_df['train'].dropna().tolist()
            val_ids = split_df['val'].dropna().tolist()
            all_patient_ids.extend(train_ids + val_ids)

            print(f"Fold {fold_idx}: {len(train_ids)} train samples, {len(val_ids)} validation samples")

            # Add to CV dictionary
            fold_name = f"Fold {fold_idx}"
            split_dict["CV"][fold_name] = {
                "Train": train_ids,
                "Val": val_ids
            }

        except Exception as e:
            print(f"Error reading {split_file}: {e}")
            continue

    patient_ids_present = np.unique(all_patient_ids)
    # Filter metadata to only include patients present in the splits
    metadata_df = metadata_df.drop_duplicates(subset=config['patient_id'])
    filtered_metadata_df = metadata_df[metadata_df[config['patient_id']].isin(patient_ids_present)]
    filtered_metadata_df.to_csv(filtered_metadata_path, index=False)

    # Save the dictionary
    ensure_directory(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(split_dict, f)



def discretize_survival_times(patients_df, label_col, censor_col, n_bins=4, eps=0.001):
    """
    Discretize survival times into bins based on quantiles of uncensored patients.

    Args:
        patients_df: DataFrame containing patient data
        label_col: Column name for survival time
        censor_col: Column name for censorship status
        n_bins: Number of bins to create
        eps: Small value to adjust bin boundaries to avoid edge issues

    Returns:
        Tuple of (updated_df, bin_boundaries)
    """
    # Make a copy to avoid modifying the original dataframe
    df = patients_df.copy()

    # Get only uncensored patients for determining quantiles
    uncensored_df = df[df[censor_col] == 0]

    if len(uncensored_df) == 0:
        print("Warning: No uncensored patients found. Using all patients for binning.")
        uncensored_df = df

    # Step 1: Determine bin boundaries using quantiles on uncensored data
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col],
                                  q=n_bins,
                                  retbins=True,
                                  labels=False)

    # Adjust bin boundaries to handle edge cases
    q_bins[-1] = df[label_col].max() + eps
    q_bins[0] = df[label_col].min() - eps

    # Step 2: Assign patients to bins using the calculated boundaries
    disc_labels, q_bins = pd.cut(df[label_col],
                                 bins=q_bins,
                                 retbins=True,
                                 labels=False,
                                 right=False,
                                 include_lowest=True)

    # Add discrete labels to patient data
    df['label'] = disc_labels.values.astype(int)

    print(f"Survival time discretized into {n_bins} bins with boundaries: {q_bins}")

    return df, q_bins


def get_survival_target(patient_row, label_col=None, censor_col=None, label_value=None):
    """
    Get target information for survival analysis.

    Args:
        patient_row: Row from dataframe containing patient data
        label_col: Column name for survival time
        censor_col: Column name for censorship status
        label_value: Pre-computed discrete label value (if available)

    Returns:
        Dictionary with target information
    """
    if label_value is not None:
        # Use pre-computed label
        label_tensor = torch.tensor(label_value, dtype=torch.long)
    else:
        # For the case where labels haven't been pre-computed
        label_tensor = torch.tensor(0, dtype=torch.long)  # Placeholder

    target_info = {
        'target': label_tensor
    }

    # Add survival-specific information if available
    if label_col is not None and censor_col is not None:
        target_info.update({
            'survival_time': patient_row[label_col],
            'censorship': torch.tensor(patient_row[censor_col], dtype=torch.long)
        })

    return target_info


def calculate_risk(outputs):
    r"""
    Take the logits of the model and calculate the risk for the patient

    Args:
        - outputs : torch.Tensor

    Returns:
        - risk : torch.Tensor

    """
    hazards = torch.sigmoid(outputs)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)

    return risk, survival


def stratify_risk_groups(risk_scores, num_groups=2):
    """
    Stratify patients into risk groups based on predicted risk scores.

    Args:
        risk_scores: numpy array of risk scores
        num_groups: number of risk groups (2 or 4)

    Returns:
        risk_groups: numpy array of risk group assignments (0 to num_groups-1)
    """
    if num_groups == 2:
        # Simple median split for 2 groups
        median_risk = np.median(risk_scores)
        risk_groups = (risk_scores > median_risk).astype(int)
    elif num_groups == 4:
        # Quartile split for 4 groups
        quartiles = np.percentile(risk_scores, [25, 50, 75])
        risk_groups = np.zeros_like(risk_scores, dtype=int)
        for i in range(1, 4):
            risk_groups[risk_scores > quartiles[i - 1]] = i
    else:
        raise ValueError("num_groups must be either 2 or 4")

    return risk_groups


def prepare_km_data(survival_times, censorships, risk_groups):
    """
    Prepare data for Kaplan-Meier analysis.

    Args:
        survival_times: numpy array of survival times
        censorships: numpy array of censorship status (0=censored, 1=event)
        risk_groups: numpy array of risk group assignments

    Returns:
        km_data: dictionary with data for each risk group
    """
    # lifelines expects event indicator as True for event occurred (opposite of your censorship)
    event_indicator = (censorships == 1)

    unique_groups = np.unique(risk_groups)
    km_data = {}

    for group in unique_groups:
        mask = (risk_groups == group)
        km_data[group] = {
            'durations': survival_times[mask],
            'event_observed': event_indicator[mask]
        }

    return km_data