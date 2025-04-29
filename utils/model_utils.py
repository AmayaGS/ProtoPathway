import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.GeneExpressionMLP import MLPBaseline
from models.GeneExprHyperGraph import BipartiteHGNN, BipartiteAttentionHGNN, HierAttnBipartiteHGNN, BipartiteGATHGNN
from models.GeneExprHyperGraph import BipartiteGAT_MHSA
from models.GeneExprHyperGraph import MLPBaseline as MLPBaselineHG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialise_model(config, input_dim):

    if config['model']['name'] == 'MLP':
        model = MLPBaseline(
            input_size=input_dim,
            hidden_size=config['training']['hidden_dim'],
            num_classes=config['n_classes'],
            dropout_rate=config['training']['dropout_rate']
        )

    if config['model']['name'] == 'Hypergraph':

        #model = MLPBaselineHG(9483, 100, 2)
        #
        # model = BipartiteHGNN(
        #     in_channels=1,  # Gene expression value
        #     hidden_channels=100,
        #     out_channels=config['n_classes'],
        #     num_layers=3,
        #     dropout=0.5
        # )
        # #
        # model = BipartiteGATHGNN(
        #     in_channels=1,  # Gene expression value
        #     hidden_channels=100,
        #     out_channels=config['n_classes'],
        #     num_layers=3,
        #     dropout=0.2
        # )

        model = BipartiteGAT_MHSA(
            in_channels=1,  # Gene expression value
            hidden_channels=100,
            out_channels=config['n_classes'],
            num_layers=3,
            dropout=0.2
        )
        # model = HierAttnBipartiteHGNN(
        #     in_channels=1,  # Gene expression value
        #     hidden_channels=100,
        #     n_classes=config['n_classes'],
        #     n_layers=3,
        #     heads=1,
        #     dropout=0.5
        #
        # )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['L2_norm'])

    sched_cfg = config.get("scheduler", {})
    if sched_cfg.get("use", False):  # default to False if not specified
        scheduler_type = sched_cfg["type"]
        # Get other params step, gamma, etc.
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    else:
        lr_scheduler = None  # No scheduler used

    if torch.cuda.is_available():
        model.cuda()

    return model, criterion, optimizer, lr_scheduler


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

        # Create cross-validation splits on the training set
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


def load_cv_folds(df, split_dict):

    training_folds = []
    validation_folds = []

    for fold_name, splits in split_dict["CV"].items():
        train_dict = {pid: df[pid] for pid in splits["Train"]}
        val_dict = {pid: df[pid] for pid in splits["Val"]}
        training_folds.append(train_dict)
        validation_folds.append(val_dict)

    return training_folds, validation_folds


def load_ge_cv_folds(df, split_dict):

    training_folds = []
    validation_folds = []

    for fold_name, splits in split_dict["CV"].items():
        train_df = df.loc[splits["Train"]]
        val_df = df.loc[splits["Val"]]
        training_folds.append(train_df)
        validation_folds.append(val_df)

    return training_folds, validation_folds

def load_ge_train_test_folds(df, split_dict):
    """Load the full training and test sets for final model training."""

    training_folds = []
    validation_folds = []

    train_df = df.loc[split_dict["Train"]]
    test_df = df.loc[split_dict["Test"]]
    training_folds.append(train_df)
    validation_folds.append(test_df)

    return training_folds, validation_folds



def load_train_test_split(data_dict, split_dict):
    """Load the full training and test sets for final model training."""
    train_dict = {pid: data_dict[pid] for pid in split_dict["Train"]}
    test_dict = {pid: data_dict[pid] for pid in split_dict["Test"]}
    return train_dict, test_dict


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
