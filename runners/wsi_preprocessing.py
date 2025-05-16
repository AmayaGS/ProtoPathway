
# runners/wsi_preprocessing.py

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import h5py

import torch

from utils.helpers import ensure_directory
from utils.survival_utils import discretize_survival_times


def wsi_preprocessing(config, experiment_logger=None):
    """
    Preprocess WSI embeddings from a directory of H5 files and save as a pickle dictionary.
    This includes integrating survival data for survival analysis tasks.

    Args:
        config: Configuration dictionary
        experiment_logger: Optional logger

    Returns:
        Path to the saved preprocessed features file
    """

    logger = experiment_logger.logger if experiment_logger else None

    if logger:
        logger.info("Starting WSI embeddings preprocessing")
    else:
        print("Starting WSI embeddings preprocessing")


    h5_dir_path = config['wsi']['input']['slides_dir']
    output_file = config['output']['data']['wsi_features']

    # Check if output file already exists
    if os.path.exists(output_file) and not config['wsi'].get('force_preprocess', False):
        if logger:
            logger.info(f"Preprocessed WSI features already exist at {output_file}")
        else:
            print(f"Preprocessed WSI features already exist at {output_file}")
        return output_file

    # Load WSI features from H5 directory
    patient_id_pattern = config.get('parsing', {}).get('patient_ID', None)
    wsi_features = load_wsi_h5_directory(config, h5_dir_path, patient_id_pattern)

    # Check if this is a survival analysis task
    is_survival = config['execution'].get('task', 'classification') == 'survival'

    if is_survival:
        if logger:
            logger.info("Integrating survival data with WSI embeddings")
        else:
            print("Integrating survival data with WSI embeddings")

        # Load patient labels with survival information
        survival_df_path = config['output']['data']['filtered_labels']

        if not os.path.exists(survival_df_path):
            raise FileNotFoundError(f"Patient labels file not found: {survival_df_path}")

        survival_df = pd.read_csv(survival_df_path)

        # Discretize survival times if needed
        if 'label' not in survival_df.columns:
            if logger:
                logger.info("Discretizing survival times")
            else:
                print("Discretizing survival times")

            survival_df, bins = discretize_survival_times(
                survival_df,
                label_col=config['survival']['target_column'],
                censor_col=config['survival']['censorship_column'],
                n_bins=config['survival']['survival_bins']
            )

        # Integrate survival data with WSI features
        patient_id_col = config['patient_id']
        time_col = config['survival']['target_column']
        event_col = config['survival']['censorship_column']

        # Modify wsi_features to include survival information
        processed_features = {}
        skipped_patients = 0

        for patient_id, (embeddings, _) in wsi_features.items():
            # Find this patient in the survival dataframe
            patient_rows = survival_df[survival_df[patient_id_col] == patient_id]

            if len(patient_rows) == 0:
                # Skip patients without survival data
                skipped_patients += 1
                continue

            # Get the patient data
            patient_row = patient_rows.iloc[0]

            # Get survival information
            survival_label = int(patient_row['label'])
            survival_time = float(patient_row[time_col])
            censorship = int(patient_row[event_col])

            # Store as tuple: (embeddings, label, patient_id, survival_time, censorship)
            processed_features[patient_id] = (
                embeddings,
                torch.tensor(survival_label, dtype=torch.long),
                patient_id,
                torch.tensor(survival_time, dtype=torch.float),
                torch.tensor(censorship, dtype=torch.long)
            )

        # Save the processed features
        with open(output_file, 'wb') as f:
            pickle.dump(processed_features, f)

        if logger:
            logger.info \
                (f"Saved {len(processed_features)} preprocessed WSI features with survival data to {output_file}")
            if skipped_patients > 0:
                logger.warning(f"Skipped {skipped_patients} patients without survival data")
        else:
            print(f"Saved {len(processed_features)} preprocessed WSI features with survival data to {output_file}")
            if skipped_patients > 0:
                print(f"Warning: Skipped {skipped_patients} patients without survival data")

    else:
        # For classification, just convert the embeddings to tensors and integrate labels
        pass

    return output_file


def load_wsi_h5_directory(config, h5_dir_path, patient_id_pattern=None):
    """
    Load WSI embeddings from a directory of H5 files (one per patient).

    Args:
        h5_dir_path: Path to directory containing H5 files
        patient_id_pattern: Optional regex pattern to extract patient ID from filename
                           If None, uses the filename without extension

    Returns:
        Dictionary mapping patient IDs to (embeddings, targets) tuples
    """
    import re

    wsi_features = {}

    h5_files = []
    for ext in ['.h5', '.hdf5']:
        h5_files.extend([os.path.join(h5_dir_path, f) for f in os.listdir(h5_dir_path)
                         if f.endswith(ext)])

    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {h5_dir_path}")

    print(f"Found {len(h5_files)} H5 files in {h5_dir_path}")

    # Get the patient ID extraction logic from config
    patient_id_extractor = config['parsing']['patient_ID']

    # Process each H5 file
    for h5_file in h5_files:
        # Extract the filename without path
        filename = os.path.basename(h5_file)

        try:
            # Use the parsing logic from config to extract patient ID
            # This evaluates the string expression from the config with 'img' set to the filename
            img = filename  # This matches the variable name used in the config
            patient_id = eval(patient_id_extractor)
            patient_id = patient_id[0] + "-" + patient_id[1] + "-" + patient_id[2] # Adjust to match the expected format

            # Load H5 file
            with h5py.File(h5_file, 'r') as f:
                # Check the structure of the file and adapt accordingly
                if 'features' in f:
                    # Direct features array
                    embeddings = f['features'][:].squeeze()
                else:
                    # Assume the first dataset in the file contains the features
                    dataset_name = list(f.keys())[0]
                    embeddings = f[dataset_name][:].squeeze()

                # Convert to tensor if needed
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)

                # Store with dummy target (will be replaced later)
                wsi_features[patient_id] = (embeddings, torch.tensor(0))

        except Exception as e:
            print(f"Error loading {h5_file}: {str(e)}")

    print(f"Successfully loaded embeddings for {len(wsi_features)} patients")

    return wsi_features