"""
Gene Expression Encoder using GATv2 on Bipartite Graph.

The bipartite graph connects genes to pathways they belong to.
GATv2 attention learns which genes are important for each pathway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import degree, softmax


# class PathwayEmbeddingModel(nn.Module):
#     """
#     Bipartite GraphSAGE model for pathway-based gene expression encoding.
#     Uses mean aggregation instead of attention to avoid collapse.
#     """
#
#     def __init__(
#             self,
#             hidden_dim=256,
#             num_layers=2,
#             dropout=0.3,
#             num_heads=1,
#             input_dim=1
#     ):
#         super().__init__()
#
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.head_dim = hidden_dim // num_heads
#
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.num_heads = num_heads
#
#         # First SAGEConv layer
#         self.conv1 = SAGEConv(
#             input_dim,
#             hidden_dim,
#             aggr='mean',
#             normalize=False,
#             project=False,
#         )
#
#         # Middle SAGEConv layers (if num_layers > 2)
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(
#                 hidden_dim,
#                 hidden_dim,
#                 aggr='mean',
#                 normalize=False,
#                 project=False,
#             ))
#
#         self.conv_final = EdgeGatedConv(hidden_dim, hidden_dim)
#
#         # # Pathway-level attention gate
#         # self.gate_nn = nn.Sequential(
#         #     nn.Linear(hidden_dim, hidden_dim // 2),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim // 2, 1)
#         # )
#
#         self._gene_attention_weights = None
#         self._attention_edge_index = None
#         self.pathway_importance = None
#
#     def forward(self, data, return_attention=False):
#         x, edge_index = data.x, data.edge_index
#         num_genes = data.num_genes
#         num_pathways = data.num_pathways
#
#         # First layer
#         x = self.conv1(x, edge_index)
#         x = F.leaky_relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Middle layers
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = F.leaky_relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Final rank-preserving layer
#         x = self.conv_final(x, edge_index, return_gates=return_attention)
#         x = F.leaky_relu(x)

        # # Extract pathway embeddings
        # pathway_embeddings = x[num_genes:num_genes + num_pathways]
        #
        # # Pathway-level attention gating
        # # gate_scores = self.gate_nn(pathway_embeddings)
        # # gate_weights = F.softmax(gate_scores, dim=0)
        #
        # # Weighted aggregation
        # # graph_embedding = (gate_weights * pathway_embeddings).mean(dim=0)
        #
        # graph_embedding = (pathway_embeddings).mean(dim=0)
        #
        #
        # return pathway_embeddings, graph_embedding

#
# class L1GATv2Conv(GATv2Conv):
#     """GATv2 with L1 normalization instead of softmax. Drop-in replacement."""
#
#     def __init__(self, *args, tau=None, center_scores=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tau = tau  # if set, use tempered softmax instead of L1
#         self.center_scores = center_scores
#
#     def message(self, x_j, x_i, index, ptr, size_i):
#         x = x_i + x_j
#         x = F.leaky_relu(x, self.negative_slope)
#         alpha = (x * self.att).sum(dim=-1)  # [num_edges, heads]
#
#         if self.tau is not None:
#             # Tempered softmax — continuous knob between softmax and uniform
#             from torch_geometric.utils import softmax
#             alpha = softmax(alpha / self.tau, index, ptr, size_i)
#         else:
#             # L1 normalization
#             if self.center_scores:
#                 score_sum = torch.zeros(size_i, self.heads, device=alpha.device)
#                 count = torch.zeros(size_i, 1, device=alpha.device)
#                 score_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha)
#                 count.scatter_add_(0, index.unsqueeze(-1),
#                                    torch.ones(index.size(0), 1, device=alpha.device))
#                 mean = score_sum / count.clamp(min=1)
#                 alpha = alpha - mean[index]
#
#             alpha = F.softplus(alpha)
#             denom = torch.zeros(size_i, self.heads, device=alpha.device)
#             denom.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha)
#             alpha = alpha / (denom[index] + 1e-8)
#
#         self._alpha = alpha.detach()
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#
#         return x_j * alpha.unsqueeze(-1)


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

        # # First GATv2 layer
        # self.conv1 = GATv2Conv(
        #     input_dim,
        #     hidden_dim,
        #     heads=num_heads,
        #     concat=False,
        #     dropout=dropout
        # )
        #
        # # Additional GATv2 layers
        # self.convs = nn.ModuleList()
        # for _ in range(num_layers - 1):
        #     self.convs.append(GATv2Conv(
        #         hidden_dim,
        #         hidden_dim,
        #         heads=num_heads,
        #         concat=False,
        #         dropout=dropout
        #     ))

        # First SAGEConv layer
        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim,
            aggr='mean',  # mean aggregation
            normalize=False,
            project=False,
        )

        # Additional SAGEConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(
                hidden_dim,
                hidden_dim,
                aggr='mean',
                normalize=False,
                project=False,
            ))

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # # Final layer
        # self.final = GATv2Conv(
        #     hidden_dim,
        #     hidden_dim,
        #     heads=num_heads,
        #     concat=False,
        #     add_self_loops=False,
        #     dropout=dropout
        # )

        # Final layer
        self.final = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=num_heads,
            concat=False,
            add_self_loops=False,
            dropout=dropout
        )

        # self.pathway_mha = nn.MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=1,  # start small
        #     dropout=0.1,
        #     batch_first=True
        # )

        # Pathway-level attention gate
        # self.gate_nn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1)
        # )

        self.gate_nn = nn.Linear(hidden_dim, 1, bias=True)

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
        x = F.leaky_relu(x)  # sparser signal with relu
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Middle layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Normalize before attention — stabilizes score magnitudes
        x = self.layer_norm(x)

        if return_attention:
            x, (edge_index_out, attn_weights) = self.final(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.final(x, edge_index)
        x = F.leaky_relu(x)

        # # Middle layers
        # attn_weights = None
        # for conv in self.convs:
        #     if return_attention:
        #         x, (edge_index_out, attn_weights) = conv(
        #             x, edge_index, return_attention_weights=True
        #         )
        #     else:
        #         x = conv(x, edge_index)
        #     x = F.leaky_relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract pathway embeddings (nodes after gene nodes)
        # gene_embeddings = x[:num_genes]

        pathway_embeddings = x[num_genes:num_genes + num_pathways]

        # gene_summary = gene_embeddings.mean(dim=0)

        # pathway_embeddings = pathway_embeddings.unsqueeze(0)  # [1, num_pathways, hidden_dim]
        # attn_out, pathway_attn_weights = self.pathway_mha(
        #     pathway_embeddings, pathway_embeddings, pathway_embeddings
        # )  # attn_out: [1, num_pathways, hidden_dim]
        # pathway_embeddings = pathway_embeddings + attn_out  # residual
        # pathway_embeddings = pathway_embeddings.squeeze(0)  # [num_pathways, hidden_dim]

        # Pathway-level attention gating
        gate_scores = self.gate_nn(pathway_embeddings)   # [num_pathways, 1]
        gate_weights = F.softmax(gate_scores, dim=0)  # [num_pathways, 1]

        # Weighted aggregation for pathway-level embedding
        graph_embedding = (gate_weights * pathway_embeddings).sum(dim=0)  # [hidden_dim]
        # graph_embedding = torch.mean(pathway_embeddings, dim=0).unsqueeze(0)

        # if self.training:
        #     self._gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum().item()
        #     self._gate_max = gate_weights.max().item()
        #     self._gate_min = gate_weights.min().item()
        #     # avg_attn = pathway_attn_weights.mean(dim=1).squeeze(0)  # [num_pathways, num_pathways]
        #     # self._mha_entropy = -(avg_attn * torch.log(avg_attn + 1e-8)).sum(dim=-1).mean().item()
        #
        # if self.training and attn_weights is not None:
        #     # attn_weights from last GATv2 layer
        #     self._gat_entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean().item()
        #     self._gat_max = attn_weights.max().item()

        # # Store attention weights if requested
        # if return_attention:
        #     self.pathway_importance = gate_weights.squeeze(-1).detach()
        #     if attn_weights is not None:
        #         self._process_attention_weights(
        #             edge_index, attn_weights, num_genes, num_pathways
        #         )

        return pathway_embeddings, graph_embedding

    # def _attention_entropy(self, weights):
    #     """High entropy = uniform, low = peaked"""
    #     return -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()

    def _process_attention_weights(self, edge_index, attn_weights, num_genes, num_pathways):
        """
        Process GATv2 attention weights into gene-pathway attention matrix.
        Also computes per-pathway attention entropy for collapse monitoring.
        """
        gene_pathway_attn = torch.zeros(num_genes, num_pathways, device=attn_weights.device)

        if attn_weights.dim() > 1:
            attn_weights = attn_weights.mean(dim=1)

        for idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
            if src < num_genes and dst >= num_genes:
                gene_idx = src
                pathway_idx = dst - num_genes
                gene_pathway_attn[gene_idx, pathway_idx] = attn_weights[idx]

        self.gene_pathway_attention = gene_pathway_attn

        # Per-pathway entropy: each row of attn per pathway
        pathway_attns = gene_pathway_attn.t()  # [num_pathways, num_genes]
        # Mask out zero entries (genes not in pathway), renormalize
        nonzero_mask = pathway_attns > 0
        counts = nonzero_mask.sum(dim=1)  # genes per pathway

        entropies = []
        for i in range(num_pathways):
            a = pathway_attns[i][nonzero_mask[i]]
            if len(a) > 1:
                a = a / (a.sum() + 1e-8)  # renormalize in case
                h = -(a * torch.log(a + 1e-8)).sum()
                entropies.append(h)

        entropies = torch.stack(entropies)
        self._pathway_entropies = entropies
        self._entropy_summary = {
            'mean': entropies.mean().item(),
            'min': entropies.min().item(),
            'max': entropies.max().item(),
            'n_collapsed': (entropies < 0.1).sum().item(),
            'n_pathways': len(entropies)
        }

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