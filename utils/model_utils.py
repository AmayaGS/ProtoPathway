import numpy as np
import pandas as pd
import pickle
import h5py
from sklearn.model_selection import StratifiedShuffleSplit

import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats


def create_cross_validation_splits(config):

    # Save everything
    output_path = os.path.join(config['output']['data']['dir'], f"data_splits_{config['dataset_name']}.pkl")
    label_path = os.path.join(config['output']['data']['dir'], f"shared_patient_labels_{config['dataset_name']}.csv")
    shared_ids_path = os.path.join(config['output']['data']['dir'], f"shared_patient_ids_{config['dataset_name']}.csv")

    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Data splits already exist at {output_path}.")
    else:
        print(f"Creating data splits and saving to {output_path}.")

        # Load the patient labels and extracted patches for the WSI dataset
        patient_labels = pd.read_csv(os.path.join(config['output']['data']['dir'], f"patient_labels_{config['dataset_name']}.csv"))
        extracted_patches = pd.read_csv(os.path.join(config['output']['data']['dir'], f"extracted_patches_level_{config['dataset']['slide_level']}_{config['dataset_name']}.csv"))
        # TODO instead of using the extracted patches, when I preprocess the WSI dataset I should save the available patient IDs/labels in a separate file.
        # This is to check the patient IDs actually in the WSI dataset
        wsi_df = pd.merge(extracted_patches, patient_labels, on=config['patient_id'])
        # Drop duplicates to obtain unique patient IDs
        df_labels = wsi_df.drop_duplicates(subset=config['patient_id']).reset_index(drop=True)

        # Load the patient IDs for the Gene Expression
        gene_expr_df = pd.read_csv(config['output']['data']['filtered_genes'], index_col=0)
        # TODO when I preprocess the gene expression dataset, I should save the available patient IDs/labels in a separate file.
        # Get common patient IDs
        wsi_patient_ids = df_labels[config['patient_id']].tolist()
        ge_patient_ids = gene_expr_df.index.tolist()
        shared_patient_ids = list(set(wsi_patient_ids) & set(ge_patient_ids))
        print(f"Found {len(shared_patient_ids)} patients with both WSI and gene expression data")

        # Filter to shared patients
        df_labels = df_labels[df_labels[config['patient_id']].isin(shared_patient_ids)].reset_index(drop=True)

        # Create a StratifiedShuffleSplit object for the test set
        X = df_labels[config['patient_id']]
        y = df_labels[config['label']]
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size= 1 - config['dataset']['train_fraction'], random_state=config['dataset']['split_seed'])
        train_val_index, test_index = next(sss_test.split(X, y))

        train_val_data = df_labels.loc[train_val_index].reset_index(drop=True)
        test_data = df_labels.loc[test_index].reset_index(drop=True)

        # Store Train and Test patient IDs
        split_dict = {
            "Train": list(train_val_data[config['patient_id']]),
            "Test": list(test_data[config['patient_id']]),
            "CV": {}
        }

        # Create cross-validation splits on the ge_training set
        sss_cv = StratifiedShuffleSplit(n_splits=config['dataset']['stratified_splits'], random_state=config['dataset']['split_seed'])

        for i, (train_idx, val_idx) in enumerate(
                sss_cv.split(train_val_data[config['patient_id']], train_val_data[config['label']])):
            fold_name = f"Fold {i}"
            split_dict["CV"][fold_name] = {
                "Train": list(train_val_data.iloc[train_idx][config['patient_id']]),
                "Val": list(train_val_data.iloc[val_idx][config['patient_id']])
            }

            # Sanity checks
            train_set = set(split_dict["CV"][fold_name]["Train"])
            val_set = set(split_dict["CV"][fold_name]["Val"])
            test_set = set(split_dict["Test"])

            assert train_set.isdisjoint(val_set), f"Train/Val overlap in {fold_name}"
            assert train_set.isdisjoint(test_set), f"Train/Test overlap in {fold_name}"
            assert val_set.isdisjoint(test_set), f"Val/Test overlap in {fold_name}"

        # Filter and save subsetted gene expression matrix
        subset_gene_expr_df = gene_expr_df.loc[shared_patient_ids]
        subset_gene_expr_path = os.path.join(config['output']['data']['dir'],
                                             f"shared_subset_gene_expression_{config['dataset_name']}.csv")
        subset_gene_expr_df.to_csv(subset_gene_expr_path)

        # Filter and save extracted patches for shared patients
        extracted_patches[config['patient_id']] = extracted_patches[config['patient_id']].astype(str)
        subset_patches_df = extracted_patches[extracted_patches[config['patient_id']].isin(shared_patient_ids)]
        subset_patches_path = os.path.join(config['output']['data']['dir'],
                                           f"shared_subset_extracted_patches_level_{config['dataset']['slide_level']}_{config['dataset_name']}.csv")
        subset_patches_df.to_csv(subset_patches_path, index=False)

        print(f"Saved subsetted gene expression data to {subset_gene_expr_path}")
        print(f"Saved subsetted extracted WSI patches to {subset_patches_path}")

        with open(output_path, "wb") as f:
            pickle.dump(split_dict, f)

        df_labels.to_csv(label_path, index=False)
        pd.DataFrame({config['patient_id']: shared_patient_ids}).to_csv(shared_ids_path, index=False)

        print(f"Saved train/test and CV splits to {output_path}")
        print(f"Saved shared patient IDs to {shared_ids_path}")
        print(f"Saved patient labels to {label_path}")


def load_gene_expression_folds(df, split_dict, is_cv=False, ignore_missing=False):
    """
    Load training and validation/test folds from a split dictionary for gene expression data.

    Args:
        df: Gene expression DataFrame with patient IDs as index
        split_dict: Dictionary with train/validation/test splits
        is_cv: Whether to load cross-validation folds (True) or train/test split (False)
        ignore_missing: If True, skip missing keys; if False, raise an error for missing keys

    Returns:
        Tuple of (training_folds, validation_folds)
    """
    training_folds = []
    validation_folds = []

    if is_cv:
        # Handle cross-validation folds
        for fold_name, splits in split_dict["CV"].items():
            if ignore_missing:
                # Filter out missing patient IDs
                train_ids = [pid for pid in splits["Train"] if pid in df.index]
                val_ids = [pid for pid in splits["Val"] if pid in df.index]

                train_df = df.loc[train_ids] if train_ids else pd.DataFrame()
                val_df = df.loc[val_ids] if val_ids else pd.DataFrame()

                if train_df.empty:
                    print(f"Warning: No valid training samples for {fold_name}")
                if val_df.empty:
                    print(f"Warning: No valid validation samples for {fold_name}")
            else:
                # Check for missing keys before creating the dataframes
                missing_train = [pid for pid in splits["Train"] if pid not in df.index]
                missing_val = [pid for pid in splits["Val"] if pid not in df.index]

                if missing_train or missing_val:
                    missing = missing_train + missing_val
                    raise KeyError(f"Patient IDs {missing} from fold {fold_name} are not present in data")

                train_df = df.loc[splits["Train"]]
                val_df = df.loc[splits["Val"]]

            training_folds.append(train_df)
            validation_folds.append(val_df)
    else:
        # Handle train/test split
        if ignore_missing:
            # Filter out missing patient IDs
            train_ids = [pid for pid in split_dict["Train"] if pid in df.index]
            test_ids = [pid for pid in split_dict["Test"] if pid in df.index]

            train_df = df.loc[train_ids] if train_ids else pd.DataFrame()
            test_df = df.loc[test_ids] if test_ids else pd.DataFrame()

            if train_df.empty:
                print(f"Warning: No valid training samples for Train/Test split")
            if test_df.empty:
                print(f"Warning: No valid test samples for Train/Test split")
        else:
            # Check for missing keys before creating the dataframes
            missing_train = [pid for pid in split_dict["Train"] if pid not in df.index]
            missing_test = [pid for pid in split_dict["Test"] if pid not in df.index]

            if missing_train or missing_test:
                missing = missing_train + missing_test
                raise KeyError(f"Patient IDs {missing} are not present in data")

            train_df = df.loc[split_dict["Train"]]
            test_df = df.loc[split_dict["Test"]]

        training_folds.append(train_df)
        validation_folds.append(test_df)

    # Report fold sizes
    for i, (train_fold, val_fold) in enumerate(zip(training_folds, validation_folds)):
        fold_name = f"Fold {i}" if is_cv else "Train/Test"
        print(f"{fold_name}: {len(train_fold)} train samples, {len(val_fold)} validation/test samples")

    return training_folds, validation_folds


def load_wsi_folds(data, split_dict, is_cv=False, ignore_missing=False):
    """
    Load training and validation/test folds from a split dictionary.

    Args:
        data: Data dictionary or DataFrame to extract folds from
        split_dict: Dictionary with train/validation/test splits
        is_cv: Whether to load cross-validation folds (True) or train/test split (False)
        ignore_missing: If True, skip missing keys; if False, raise an error for missing keys

    Returns:
        Tuple of (training_folds, validation_folds)
    """
    training_folds = []
    validation_folds = []

    if is_cv:
        # Handle cross-validation folds
        for fold_name, splits in split_dict["CV"].items():
            # Create train and validation dictionaries using dict comprehensions
            if ignore_missing:
                train_data = {pid: data[pid] for pid in splits["Train"] if pid in data}
                val_data = {pid: data[pid] for pid in splits["Val"] if pid in data}
            else:
                # Check for missing keys before creating the dictionaries
                missing_train = [pid for pid in splits["Train"] if pid not in data]
                missing_val = [pid for pid in splits["Val"] if pid not in data]

                if missing_train or missing_val:
                    missing = missing_train + missing_val
                    raise KeyError(f"Keys {missing} from fold {fold_name} are not present in data")

                train_data = {pid: data[pid] for pid in splits["Train"]}
                val_data = {pid: data[pid] for pid in splits["Val"]}

            training_folds.append(train_data)
            validation_folds.append(val_data)
    else:
        # Handle train/test split
        if ignore_missing:
            train_data = {k: data[k] for k in split_dict['Train'] if k in data}
            test_data = {k: data[k] for k in split_dict['Test'] if k in data}
        else:
            # Check for missing keys before creating the dictionaries
            missing_train = [k for k in split_dict['Train'] if k not in data]
            missing_test = [k for k in split_dict['Test'] if k not in data]

            if missing_train or missing_test:
                missing = missing_train + missing_test
                raise KeyError(f"Keys {missing} are not present in data")

            train_data = {k: data[k] for k in split_dict['Train']}
            test_data = {k: data[k] for k in split_dict['Test']}

        training_folds.append(train_data)
        validation_folds.append(test_data)

    return training_folds, validation_folds


def minority_sampler(dataset):

    # Get labels
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]['target'].item())

    # Count class occurrences
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count

    # Create sample weights
    weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(weights)

    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def l1_regularization(model, l1_norm):
    weights = sum(torch.abs(p).sum() for p in model.parameters())
    return weights * l1_norm


def _seed_torch(seed=42, device='cuda'):

    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#
# def load_cv_folds(df, split_dict):
#
#     training_folds = []
#     validation_folds = []
#
#     for fold_name, splits in split_dict["CV"].items():
#         train_dict = {pid: df[pid] for pid in splits["Train"]}
#         val_dict = {pid: df[pid] for pid in splits["Val"]}
#         training_folds.append(train_dict)
#         validation_folds.append(val_dict)
#
#     return training_folds, validation_folds
#
# def load_wsi_train_test_folds(data_dict, split_dict):
#
#     training_folds = []
#     validation_folds = []
#
#     train_dict = {k: data_dict[k] for k in split_dict['Train']}
#     test_dict = {k: data_dict[k] for k in split_dict['Test']}
#     training_folds.append(train_dict)
#     validation_folds.append(test_dict)
#
#     return training_folds, validation_folds
#
#
# def load_ge_cv_folds(df, split_dict):
#
#     training_folds = []
#     validation_folds = []
#
#     for fold_name, splits in split_dict["CV"].items():
#         train_df = df.loc[splits["Train"]]
#         val_df = df.loc[splits["Val"]]
#         training_folds.append(train_df)
#         validation_folds.append(val_df)
#
#     return training_folds, validation_folds
#
# def load_ge_train_test_folds(df, split_dict):
#     """Load the full ge_training and test sets for final model ge_training."""
#
#     training_folds = []
#     validation_folds = []
#
#     train_df = df.loc[split_dict["Train"]]
#     test_df = df.loc[split_dict["Test"]]
#     training_folds.append(train_df)
#     validation_folds.append(test_df)
#
#     return training_folds, validation_folds
#
#
# def load_train_test_split(data_dict, split_dict):
#     """Load the full ge_training and test sets for final model ge_training."""
#     train_dict = {pid: data_dict[pid] for pid in split_dict["Train"]}
#     test_dict = {pid: data_dict[pid] for pid in split_dict["Test"]}
#     return train_dict, test_dict

#
# def load_wsi_folds(data, split_dict, is_cv=True):
#
#     training_folds = []
#     validation_folds = []
#
#     if is_cv:
#         # Handle cross-validation folds
#         for fold_name, splits in split_dict["CV"].items():
#             train_data = {pid: data[pid] for pid in splits["Train"]}
#             val_data = {pid: data[pid] for pid in splits["Val"]}
#             training_folds.append(train_data)
#             validation_folds.append(val_data)
#     else:
#         # Handle train/test split
#         train_data = {k: data[k] for k in split_dict['Train']}
#         test_data = {k: data[k] for k in split_dict['Test']}
#         training_folds.append(train_data)
#         validation_folds.append(test_data)
#
#     return training_folds, validation_folds

# def initialise_model(config, input_dim):
#
#     if config['gene_expression']['model'] == 'MLP':
#         model = MLPBaseline(
#             input_size=input_dim,
#             hidden_size=config['ge_training']['hidden_dim'],
#             num_classes=config['num_classes'],
#             dropout_rate=config['ge_training']['dropout_rate']
#         )
#
#     if config['gene_expression']['model'] == 'Hypergraph':
#
#         model = PathwayEmbeddingModel(config, in_channels=1, hidden_channels=config['ge_training']['hidden_dim'],
#                                       out_channels=config['num_classes'], num_layers=3, dropout=0.2)
#
#         # model = BipartiteGAT_MHSA(
#         #     in_channels=1,  # Gene expression value
#         #     hidden_channels=100,
#         #     out_channels=config['num_classes'],
#         #     num_layers=3,
#         #     dropout=0.2
#         # )
#
#         #model = MLPBaselineHG(9483, 100, 2)
#
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=config['ge_training']['learning_rate'], weight_decay=config['ge_training']['L2_norm'])
#
#     sched_cfg = config.get("scheduler", {})
#     if sched_cfg.get("use", False):  # default to False if not specified
#         scheduler_type = sched_cfg["type"]
#         # Get other params step, gamma, etc.
#         lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
#     else:
#         lr_scheduler = None  # No scheduler used
#
#     if torch.cuda.is_available():
#         model.cuda()
#
#     return model, criterion, optimizer, lr_scheduler