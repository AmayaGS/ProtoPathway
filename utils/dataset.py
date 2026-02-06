"""
Multimodal Dataset for ProtoPathway.

Handles loading and combining:
- Gene expression data (with bipartite graph structure)
- WSI patch features
- Survival/classification labels

Designed to work with preprocessed outputs from the pipeline.
"""

import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MultimodalDataset(Dataset):
    """
    Unified multimodal dataset for ProtoPathway.

    Loads gene expression, WSI features, and labels for a set of patients.
    Returns PyTorch Geometric Data objects for the bipartite graph.
    """

    def __init__(
            self,
            patient_ids,
            gene_expression_df,
            graph_data,
            wsi_features,
            labels_df,
            task='survival',
            patient_id_col='case_id',
            return_patient_id=True
    ):
        """
        Initialize the dataset.

        Args:
            patient_ids: List of patient IDs to include
            gene_expression_df: DataFrame with patients as rows, genes as columns
            graph_data: Dict from build_bipartite_graph (edge_index, gene_names, etc.)
            wsi_features: Dict {patient_id: tensor of shape [num_patches, feature_dim]}
            labels_df: DataFrame with patient_id and label columns
            task: 'survival' or 'classification'
            patient_id_col: Column name for patient ID in labels_df
            return_patient_id: Whether to include patient_id in returned data
        """
        self.task = task
        self.patient_id_col = patient_id_col
        self.return_patient_id = return_patient_id

        # Store graph structure (shared across all samples)
        self.edge_index = graph_data['edge_index']
        self.num_genes = graph_data['num_genes']
        self.num_pathways = graph_data['num_pathways']
        self.gene_names = graph_data['gene_names']
        self.pathway_names = graph_data['pathway_names']
        self.pathway_gene_indices = graph_data['pathway_gene_indices']

        # Filter to patients with all modalities
        gene_patients = set(str(pid) for pid in gene_expression_df.index)
        wsi_patients = set(str(pid) for pid in wsi_features.keys())
        label_patients = set(str(pid) for pid in labels_df[patient_id_col])
        requested_patients = set(str(pid) for pid in patient_ids)

        # Intersection of all sources
        self.patient_ids = sorted(list(
            requested_patients & gene_patients & wsi_patients & label_patients
        ))

        if len(self.patient_ids) < len(patient_ids):
            missing = len(patient_ids) - len(self.patient_ids)
            logging.warning(f"Missing data for {missing} patients, using {len(self.patient_ids)}")

        # Store data references
        self.gene_expression_df = gene_expression_df[self.gene_names]
        self.wsi_features = wsi_features

        assert list(self.gene_expression_df.columns) == self.gene_names, \
            "Gene expression columns don't match graph gene names"

        # Build label lookup
        self.labels_df = labels_df.copy()
        self.labels_df[patient_id_col] = self.labels_df[patient_id_col].astype(str)
        self.labels_df = self.labels_df.set_index(patient_id_col)

        logging.info(f"Dataset initialized: {len(self.patient_ids)} patients, "
                     f"{self.num_genes} genes, {self.num_pathways} pathways")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # --- Gene Expression (Bipartite Graph) ---
        # Gene expression values as node features
        gene_expr = self.gene_expression_df.loc[patient_id].values

        # Debug check
        if len(gene_expr) != self.num_genes:
            raise ValueError(
                f"Patient {patient_id} (idx={idx}): gene expression has {len(gene_expr)} genes, "
                f"expected {self.num_genes}. Check if this patient exists in filtered data."
            )

        gene_expr_tensor = torch.tensor(gene_expr, dtype=torch.float32).reshape(-1, 1)

        # Create node features: genes get expression values, pathways get zeros
        x = torch.zeros((self.num_genes + self.num_pathways, 1), dtype=torch.float32)
        x[:self.num_genes] = gene_expr_tensor

        # --- WSI Features ---
        wsi_feat = self.wsi_features[patient_id]
        if not isinstance(wsi_feat, torch.Tensor):
            wsi_feat = torch.tensor(wsi_feat, dtype=torch.float32)

        # --- Labels ---
        label_row = self.labels_df.loc[patient_id]

        if self.task == 'survival':
            target = {
                'bin': torch.tensor(int(label_row['survival_bin']), dtype=torch.long),
                'time': torch.tensor(float(label_row['survival_time']), dtype=torch.float32),
                'event': torch.tensor(int(label_row['event']), dtype=torch.long)
            }
        else:
            target = torch.tensor(int(label_row['label']), dtype=torch.long)

        # --- Build PyG Data Object ---
        data = Data(
            x=x,
            edge_index=self.edge_index,
            num_genes=self.num_genes,
            num_pathways=self.num_pathways
        )

        # Add other fields
        data.wsi_features = wsi_feat
        data.y = target
        data.pathway_gene_indices = self.pathway_gene_indices

        if self.return_patient_id:
            data.patient_id = patient_id

        return data


def load_dataset_components(cfg):
    """
    Load all dataset components from preprocessed files.

    Args:
        cfg: OmegaConf config with input paths

    Returns:
        dict with gene_expression_df, graph_data, wsi_features, labels_df, splits
    """
    logging.info("Loading dataset components...")

    # Gene expression
    gene_expression_df = pd.read_csv(cfg.input.gene_expression, index_col=0)
    gene_expression_df.index = gene_expression_df.index.astype(str)

    if gene_expression_df.index.duplicated().any():
        n_dups = gene_expression_df.index.duplicated().sum()
        logging.warning(f"Found {n_dups} duplicate patient IDs in gene expression, keeping first occurrence")
        gene_expression_df = gene_expression_df[~gene_expression_df.index.duplicated(keep='first')]

    logging.info(f"  Gene expression: {gene_expression_df.shape}")

    # Bipartite graph
    graph_data = torch.load(cfg.input.bipartite_graph, weights_only=False)

    # Precompute pathway->gene indices for baselines
    num_genes = graph_data['num_genes']
    edge_index = graph_data['edge_index']
    mask = edge_index[1] >= num_genes
    gene_idx = edge_index[0][mask]
    pathway_idx = edge_index[1][mask] - num_genes

    pathway_gene_indices = [[] for _ in range(graph_data['num_pathways'])]
    for g, p in zip(gene_idx.tolist(), pathway_idx.tolist()):
        pathway_gene_indices[p].append(g)
    graph_data['pathway_gene_indices'] = pathway_gene_indices

    logging.info(f"  Graph: {graph_data['num_genes']} genes, {graph_data['num_pathways']} pathways")

    # WSI features
    with open(cfg.input.wsi_features, 'rb') as f:
        wsi_features = pickle.load(f)
    # Ensure keys are strings
    wsi_features = {str(k): v for k, v in wsi_features.items()}
    logging.info(f"  WSI features: {len(wsi_features)} patients")

    # Labels
    labels_df = pd.read_csv(cfg.input.labels)
    logging.info(f"  Labels: {len(labels_df)} patients")

    # Splits
    with open(cfg.input.splits, 'rb') as f:
        splits = pickle.load(f)
    logging.info(f"  Splits: {len(splits['CV'])} folds")

    return {
        'gene_expression_df': gene_expression_df,
        'graph_data': graph_data,
        'wsi_features': wsi_features,
        'labels_df': labels_df,
        'splits': splits
    }


def create_fold_datasets(data_components, fold_splits, cfg):
    """
    Create train and validation datasets for a single fold.

    Args:
        data_components: Dict from load_dataset_components
        fold_splits: Dict with 'Train' and 'Val' patient ID lists
        cfg: Config object

    Returns:
        train_dataset, val_dataset
    """
    train_dataset = MultimodalDataset(
        patient_ids=fold_splits['Train'],
        gene_expression_df=data_components['gene_expression_df'],
        graph_data=data_components['graph_data'],
        wsi_features=data_components['wsi_features'],
        labels_df=data_components['labels_df'],
        task=cfg.task,
        patient_id_col=cfg.patient_id_col
    )

    val_dataset = MultimodalDataset(
        patient_ids=fold_splits['Val'],
        gene_expression_df=data_components['gene_expression_df'],
        graph_data=data_components['graph_data'],
        wsi_features=data_components['wsi_features'],
        labels_df=data_components['labels_df'],
        task=cfg.task,
        patient_id_col=cfg.patient_id_col
    )

    return train_dataset, val_dataset


def sample_wsi_embeddings(wsi_features, max_samples=100000, seed=42):
    """
    Sample patch embeddings from all patients for k-means initialization.

    Args:
        wsi_features: Dict {patient_id: tensor [num_patches, feature_dim]}
        max_samples: Maximum patches to sample
        seed: Random seed

    Returns:
        torch.Tensor of shape [num_samples, feature_dim]
    """
    np.random.seed(seed)

    tensors = list(wsi_features.values())
    patch_counts = torch.tensor([t.shape[0] for t in tensors])
    total_patches = int(patch_counts.sum())

    if total_patches <= max_samples:
        keep_counts = patch_counts
    else:
        ratio = max_samples / total_patches
        keep_counts = torch.clamp((patch_counts * ratio).long(), min=1)

    sampled = []
    for t, k in zip(tensors, keep_counts.tolist()):
        idx = np.random.choice(len(t), size=min(k, len(t)), replace=False)
        sampled.append(t[idx])

    out = torch.cat(sampled)

    if len(out) > max_samples:
        idx = np.random.choice(len(out), max_samples, replace=False)
        out = out[idx]

    logging.info(f"Sampled {len(out)} patches from {len(tensors)} patients")
    return out


def compute_centroids(embeddings, n_clusters, seed=42, cosine=True):
    """
    Compute k-means centroids for prototype initialization.

    Args:
        embeddings: Tensor [num_samples, feature_dim]
        n_clusters: Number of centroids
        seed: Random seed
        cosine: Use cosine similarity (L2 normalize before clustering)

    Returns:
        torch.Tensor [n_clusters, feature_dim]
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    X = embeddings.cpu().numpy()

    if cosine:
        X = normalize(X)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,
        random_state=seed,
        n_init=10
    ).fit(X)

    C = km.cluster_centers_

    if cosine:
        C = normalize(C)

    logging.info(f"Computed {n_clusters} centroids")
    return torch.tensor(C, dtype=torch.float32)