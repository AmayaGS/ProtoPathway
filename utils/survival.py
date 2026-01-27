"""
Survival analysis utilities.

Functions for:
- Discretizing continuous survival times into risk bins
- Computing survival targets for training
- Handling censorship
"""

import logging
import numpy as np


def discretize_survival_times(df, time_col, event_col, n_bins=4, method='quantile'):
    """
    Discretize continuous survival times into risk bins.

    Uses quantile-based binning on uncensored patients to determine bin edges,
    then assigns all patients (including censored) to bins.

    Args:
        df: DataFrame with survival data
        time_col: Column name for survival time
        event_col: Column name for event indicator (1=event, 0=censored)
        n_bins: Number of risk bins
        method: 'quantile' for quantile-based bins, 'uniform' for uniform bins

    Returns:
        df: DataFrame with added 'survival_bin' column
        bin_edges: Array of bin edges used
    """
    df = df.copy()

    times = df[time_col].values
    events = df[event_col].values

    # Get uncensored times for determining bin edges
    uncensored_mask = events == 1
    uncensored_times = times[uncensored_mask]

    if len(uncensored_times) < n_bins:
        logging.warning(f"Only {len(uncensored_times)} uncensored samples, "
                        f"using all samples for binning")
        uncensored_times = times

    # Compute bin edges
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(uncensored_times, percentiles)
    else:  # uniform
        bin_edges = np.linspace(uncensored_times.min(), uncensored_times.max(), n_bins + 1)

    # Ensure unique bin edges (can happen with ties)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < n_bins + 1:
        logging.warning(f"Reduced to {len(bin_edges) - 1} bins due to ties in survival times")

    # Assign all patients to bins
    # np.digitize returns 1-indexed bins, subtract 1 for 0-indexed
    survival_bins = np.digitize(times, bin_edges[1:-1])  # n_bins - 1 internal edges

    # Clip to valid range (in case of edge cases)
    survival_bins = np.clip(survival_bins, 0, len(bin_edges) - 2)

    df['survival_bin'] = survival_bins

    # Log distribution
    bin_counts = df['survival_bin'].value_counts().sort_index()
    logging.info(f"Survival bin distribution (n_bins={len(bin_edges) - 1}):")
    for bin_idx, count in bin_counts.items():
        if bin_idx < len(bin_edges) - 1:
            logging.info(f"  Bin {bin_idx}: {count} patients "
                         f"(time: {bin_edges[bin_idx]:.1f} - {bin_edges[bin_idx + 1]:.1f})")

    return df, bin_edges


def get_survival_label_dict(df, patient_id_col, time_col, event_col, n_bins=4):
    """
    Create a dictionary mapping patient IDs to survival information.

    Args:
        df: DataFrame with survival data
        patient_id_col: Column name for patient ID
        time_col: Column name for survival time
        event_col: Column name for event indicator
        n_bins: Number of risk bins for discretization

    Returns:
        dict: {patient_id: {
            'time': float,
            'event': int,
            'bin': int
        }}
        bin_edges: Array of bin edges used
    """
    # Discretize
    df, bin_edges = discretize_survival_times(df, time_col, event_col, n_bins)

    # Build dictionary
    label_dict = {}
    for _, row in df.iterrows():
        patient_id = str(row[patient_id_col])
        label_dict[patient_id] = {
            'time': float(row[time_col]),
            'event': int(row[event_col]),
            'bin': int(row['survival_bin'])
        }

    return label_dict, bin_edges


def get_classification_label_dict(df, patient_id_col, label_col):
    """
    Create a dictionary mapping patient IDs to classification labels.

    Args:
        df: DataFrame with label data
        patient_id_col: Column name for patient ID
        label_col: Column name for classification label

    Returns:
        dict: {patient_id: int label}
    """
    label_dict = {}
    for _, row in df.iterrows():
        patient_id = str(row[patient_id_col])
        label_dict[patient_id] = int(row[label_col])

    return label_dict


def compute_survival_statistics(times, events):
    """
    Compute basic survival statistics.

    Args:
        times: Array of survival times
        events: Array of event indicators

    Returns:
        dict with statistics
    """
    times = np.array(times)
    events = np.array(events)

    uncensored_mask = events == 1

    stats = {
        'n_total': len(times),
        'n_events': int(uncensored_mask.sum()),
        'n_censored': int((~uncensored_mask).sum()),
        'event_rate': float(uncensored_mask.mean()),
        'median_time_all': float(np.median(times)),
        'median_time_events': float(np.median(times[uncensored_mask])) if uncensored_mask.any() else None,
        'min_time': float(times.min()),
        'max_time': float(times.max()),
    }

    return stats