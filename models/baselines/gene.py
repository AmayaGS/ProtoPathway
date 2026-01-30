"""
Gene Expression Baseline Models for ProtoPathway.

Implements:
- SNN: Survival Neural Network (works with bipartite graph structure)
- MLP: Simple Multi-Layer Perceptron (flattened gene expression)

All models follow the same interface:
    logits = model(data)  # data.x: node features, data.edge_index: graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool


class SNN(nn.Module):
    """
    Survival Neural Network for gene expression.

    This model works with the bipartite graph structure but uses a simpler
    architecture than ProtoPathway's GATv2-based pathway embedding.

    Architecture:
        gene_features -> pathway_aggregation -> MLP -> survival_logits

    Reference:
        Katzman et al. "DeepSurv" (BMC Medical Research Methodology 2018)
        Extended for discrete survival bins (NLL loss compatible)
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dims: list = [256, 128],
        n_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_genes = num_genes

        # Gene embedding layer
        self.gene_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0] // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Build MLP layers (LayerNorm for batch_size=1 compatibility)
        layers = []
        in_dim = num_genes * (hidden_dims[0] // 4)  # Flattened gene embeddings

        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SELU())  # SELU for self-normalizing
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)

        # Weight initialization for SELU
        self._init_weights()

        # Storage for interpretability
        self._gene_importance = None

    def _init_weights(self):
        """Initialize weights for SELU activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: [num_nodes, 1] node features (genes + pathways)
                - num_genes: number of gene nodes
            return_attention: Whether to compute gene importance

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
                sample_x = x[mask][:num_genes]  # [num_genes, 1]
                gene_features_list.append(sample_x)

            gene_features = torch.stack(gene_features_list)  # [B, num_genes, 1]
        else:
            gene_features = x[:num_genes].unsqueeze(0)  # [1, num_genes, 1]

        B, G, _ = gene_features.shape

        # Embed each gene
        h = self.gene_embed(gene_features)  # [B, num_genes, hidden//4]

        # Flatten
        h = h.view(B, -1)  # [B, num_genes * hidden//4]

        # MLP
        h = self.mlp(h)  # [B, hidden_dims[-1]]

        # Classifier
        logits = self.classifier(h)  # [B, n_classes]

        if return_attention:
            # Compute gradient-based gene importance
            self._gene_importance = gene_features.detach()

        return logits

    def get_attention_outputs(self):
        """Return gene importance scores."""
        return {'gene_importance': self._gene_importance}


class GeneExpressionMLP(nn.Module):
    """
    Simple MLP baseline for gene expression classification/survival.

    Takes flattened gene expression as input (ignores graph structure).

    Architecture:
        gene_expression_vector -> MLP -> logits
    """

    def __init__(
        self,
        num_genes: int,
        hidden_dims: list = [512, 256, 128],
        n_classes: int = 4,
        dropout: float = 0.3
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

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def get_attention_outputs(self):
        """No attention in MLP."""
        return {}


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
        hidden_dims: list = [256, 128],
        n_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_pathways = num_pathways

        # Gene projection
        self.gene_proj = nn.Linear(1, 32)

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
        gene_h = self.gene_proj(gene_x)  # [num_genes, 32]

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