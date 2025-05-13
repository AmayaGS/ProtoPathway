
# utils/survival_utils.py

import pandas as pd
import numpy as np
import torch


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
            'censorship': patient_row[censor_col]
        })

    return target_info


def calculate_c_index(survival_times, censorships, risk_scores):
    """
    Calculate the concordance index (c-index) for survival analysis.

    Args:
        survival_times: numpy array of survival times
        censorships: numpy array of censorship status (0=uncensored, 1=censored)
        risk_scores: numpy array of risk scores (higher score = higher risk)

    Returns:
        concordance index (float)
    """
    try:
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        raise ImportError("sksurv library is required for survival analysis. Please install it.")

    # sksurv expects event_indicator as True for uncensored (event occurred)
    # and False for censored
    event_indicator = ~censorships.astype(bool)  # Convert 0=uncensored to True

    # Get concordance index and other statistics
    concordance, concordant_pairs, discordant_pairs, tied_risk, tied_time = concordance_index_censored(
        event_indicator,
        survival_times,
        risk_scores
    )

    return concordance