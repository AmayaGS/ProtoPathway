"""
Gene Expression Encoder using GATv2 on Bipartite Graph.

The bipartite graph connects genes to pathways they belong to.
GATv2 attention learns which genes are important for each pathway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class PathwayEmbeddingModel(nn.Module):
    """
    Bipartite GATv2 model for pathway-based gene expression encoding.

    Architecture:
    1. GATv2 layers propagate information between genes and pathways
    2. Attention weights capture gene importance within pathways
    3. Pathway-level gating produces final embedding

    Input graph structure:
    - Nodes 0 to num_genes-1: gene nodes (features = expression values)
    - Nodes num_genes to num_genes+num_pathways-1: pathway nodes (features = 0)
    - Bidirectional edges connect genes to their pathways
    """

    def __init__(
            self,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            num_heads=1,
            input_dim=1
    ):
        """
        Initialize the pathway embedding model.

        Args:
            hidden_dim: Hidden dimension for GATv2 layers
            num_layers: Number of GATv2 layers
            dropout: Dropout rate
            num_heads: Number of attention heads (output is concatenated then projected)
            input_dim: Input feature dimension (1 for scalar expression values)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # First GATv2 layer
        self.conv1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout
        )

        # Additional GATv2 layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout
            ))
        #
        # self.pathway_mha = nn.MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=1,  # start small
        #     dropout=0.1,
        #     batch_first=True
        # )

        # Pathway-level attention gate
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Storage for attention weights (populated during forward if requested)
        self.gene_pathway_attention = None
        self.pathway_importance = None

    def forward(self, data, return_attention=False):
        """
        Forward pass through the pathway embedding model.

        Args:
            data: PyG Data object with:
                - x: Node features [num_genes + num_pathways, input_dim]
                - edge_index: Bipartite edges [2, num_edges]
                - num_genes: Number of gene nodes
                - num_pathways: Number of pathway nodes
            return_attention: If True, store attention weights for visualization

        Returns:
            pathway_embeddings: [num_pathways, hidden_dim] - individual pathway embeddings
            graph_embedding: [hidden_dim] - aggregated graph-level embedding
            (attention weights are stored in self.gene_pathway_attention if requested)
        """
        x, edge_index = data.x, data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        # First layer
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        attn_weights = None
        for conv in self.convs:
            if return_attention:
                x, (edge_index_out, attn_weights) = conv(
                    x, edge_index, return_attention_weights=True
                )
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract pathway embeddings (nodes after gene nodes)
        pathway_embeddings = x[num_genes:num_genes + num_pathways]  # [num_pathways, hidden_dim]

        # pathway_embeddings = pathway_embeddings.unsqueeze(0)  # [1, num_pathways, hidden_dim]
        # attn_out, pathway_attn_weights = self.pathway_mha(
        #     pathway_embeddings, pathway_embeddings, pathway_embeddings
        # )  # attn_out: [1, num_pathways, hidden_dim]
        # pathway_embeddings = pathway_embeddings + attn_out  # residual
        # pathway_embeddings = pathway_embeddings.squeeze(0)  # [num_pathways, hidden_dim]

        # Pathway-level attention gating
        gate_scores = self.gate_nn(pathway_embeddings)  # [num_pathways, 1]
        gate_weights = F.softmax(gate_scores, dim=0)  # [num_pathways, 1]

        # Weighted aggregation for pathway-level embedding
        graph_embedding = (gate_weights * pathway_embeddings).mean(dim=0)  # [hidden_dim]
        # graph_embedding = torch.mean(pathway_embeddings, dim=0).unsqueeze(0)

        # Store attention weights if requested
        if return_attention:
            self.pathway_importance = gate_weights.squeeze(-1).detach()
            if attn_weights is not None:
                self._process_attention_weights(
                    edge_index, attn_weights, num_genes, num_pathways
                )

        return pathway_embeddings, graph_embedding

    def _process_attention_weights(self, edge_index, attn_weights, num_genes, num_pathways):
        """
        Process GATv2 attention weights into gene-pathway attention matrix.

        Creates a [num_genes, num_pathways] matrix where entry (i,j) is the
        attention weight from gene i to pathway j.
        """
        # Initialize attention matrix
        gene_pathway_attn = torch.zeros(num_genes, num_pathways, device=attn_weights.device)

        # attn_weights: [num_edges, num_heads] - we average across heads
        if attn_weights.dim() > 1:
            attn_weights = attn_weights.mean(dim=1)

        # Process edges: gene -> pathway edges
        for idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()

            # Only count gene -> pathway edges
            if src < num_genes and dst >= num_genes:
                gene_idx = src
                pathway_idx = dst - num_genes
                gene_pathway_attn[gene_idx, pathway_idx] = attn_weights[idx]

        self.gene_pathway_attention = gene_pathway_attn

    def get_gene_importance(self):
        """
        Get overall gene importance scores (aggregated across pathways).

        Returns:
            Tensor [num_genes] with importance scores
        """
        if self.gene_pathway_attention is None or self.pathway_importance is None:
            return None

        # Weight gene-pathway attention by pathway importance
        weighted_attn = self.gene_pathway_attention * self.pathway_importance.unsqueeze(0)
        gene_importance = weighted_attn.sum(dim=1)

        return gene_importance

    def get_pathway_importance(self):
        """Get pathway importance scores from gating mechanism."""
        return self.pathway_importance


class GeneEncoderUnimodal(nn.Module):
    """
    Standalone gene encoder for unimodal gene expression prediction.

    Wraps PathwayEmbeddingModel and adds a classification/survival head.
    """

    def __init__(
            self,
            num_classes,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            num_heads=1
    ):
        super().__init__()

        self.encoder = PathwayEmbeddingModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, return_attention=False):
        """
        Forward pass for unimodal prediction.

        Returns:
            logits: [num_classes] prediction logits
        """
        _, graph_embedding = self.encoder(data, return_attention=return_attention)
        logits = self.classifier(graph_embedding.unsqueeze(0))  # [1, num_classes]
        return logits