"""
Gene Expression Baseline Models for ProtoPathway.

Implements:
- SNN: Self Normalising Neural Network
- MLP: Simple Multi-Layer Perceptron

Take as input the bipartite graph and extract the gene expression

All models follow the same interface:
    logits = model(data)  # data.x: node features, data.edge_index: graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


class SNN(nn.Module):
    """
Implement a self normalizing network to handle tabular omics data

Klambauer, Günter, et al.
"Self-normalizing neural networks." Advances in neural information processing systems 30 (2017).
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dims: list = [256, 256],
        n_classes: int = 4,
        dropout: float = 0.25
    ):
        super().__init__()

        layers = []
        in_dim = num_genes

        self.num_genes = num_genes

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout, inplace=False))
            in_dim = out_dim

        self.snn = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dims[-1], n_classes)


    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: [num_nodes, 1] node features (genes + pathways)
                - num_genes: number of gene nodes

        Returns:
            logits: [batch_size, n_classes]
        """
        # Extract gene features only (first num_genes nodes)
        x = data.x
        num_genes = data.num_genes if hasattr(data, 'num_genes') else self.num_genes

        # Handle batched data
        if hasattr(data, 'batch'):
            batch = data.batch
            batch_size = batch.max().item() + 1

            # Separate gene features per sample
            gene_features_list = []
            for b in range(batch_size):
                mask = batch == b
                # Get first num_genes nodes for this sample
                sample_x = x[mask][:num_genes].squeeze(-1)  # [num_genes]
                gene_features_list.append(sample_x)

            gene_features = torch.stack(gene_features_list)  # [B, num_genes, 1]
        else:
            gene_features = x[:num_genes].squeeze(-1).unsqueeze(0)  # [1, num_genes]

        h = self.snn(gene_features)
        logits = self.classifier(h)

        return logits



class GeneExpressionMLP(nn.Module):
    """
    Simple MLP baseline for gene expression classification/survival.

    Takes gene expression as input.

    Architecture:
        gene_expression_vector -> MLP -> logits
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dims: list = [256, 256],
        n_classes: int = 4,
        dropout: float = 0.25
    ):
        super().__init__()

        self.num_genes = num_genes

        layers = []
        in_dim = num_genes

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data object with x: [num_nodes, 1]

        Returns:
            logits: [batch_size, n_classes]
        """
        x = data.x
        num_genes = data.num_genes if hasattr(data, 'num_genes') else self.num_genes

        # Handle batched data
        if hasattr(data, 'batch'):
            batch = data.batch
            batch_size = batch.max().item() + 1

            gene_features_list = []
            for b in range(batch_size):
                mask = batch == b
                sample_x = x[mask][:num_genes].squeeze(-1)  # [num_genes]
                gene_features_list.append(sample_x)

            gene_features = torch.stack(gene_features_list)  # [B, num_genes]
        else:
            gene_features = x[:num_genes].squeeze(-1).unsqueeze(0)  # [1, num_genes]

        # MLP forward
        h = self.mlp(gene_features)
        logits = self.classifier(h)

        return logits



class PathwayMLP(nn.Module):
    """
    MLP that aggregates genes by pathway before classification.

    Uses the bipartite graph structure to group genes into pathways,
    then applies MLP on pathway-level features.

    Architecture:
        genes -> pathway_aggregation (mean) -> MLP -> logits
    """

    def __init__(
        self,
        num_genes: int,
        num_pathways: int,
        hidden_dims: list = [256, 256],
        n_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_pathways = num_pathways

        # Gene projection
        self.gene_proj = nn.Linear(1, 128)

        # Pathway MLP (after aggregation)
        layers = []
        in_dim = num_pathways * 32

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, data, return_attention: bool = False):
        """Forward pass with pathway aggregation."""
        x = data.x
        edge_index = data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        # Project gene features
        gene_x = x[:num_genes]  # [num_genes, 1]
        gene_h = self.gene_proj(gene_x)  # [num_genes, 128]

        # Aggregate genes to pathways using edge_index
        # edge_index[0] = gene indices, edge_index[1] = pathway indices
        pathway_h = torch.zeros(num_pathways, gene_h.size(1), device=gene_h.device)
        counts = torch.zeros(num_pathways, device=gene_h.device)

        gene_idx = edge_index[0]
        pathway_idx = edge_index[1] - num_genes  # Adjust to 0-indexed

        pathway_h.index_add_(0, pathway_idx, gene_h[gene_idx])
        counts.index_add_(0, pathway_idx, torch.ones_like(pathway_idx, dtype=torch.float))

        pathway_h = pathway_h / counts.clamp(min=1).unsqueeze(1)  # Mean pooling

        # Flatten and classify
        h = pathway_h.view(1, -1)  # [1, num_pathways * 32]
        h = self.mlp(h)
        logits = self.classifier(h)

        return logits

    def get_attention_outputs(self):
        return {}