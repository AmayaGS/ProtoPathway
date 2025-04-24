

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import HypergraphConv, GlobalAttention

from torch.sparse import FloatTensor

class BipartiteHGNN(torch.nn.Module):
    """
    Bipartite graph representation of HGNN for gene expression data.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(BipartiteHGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers for the bipartite representation
        self.conv1 = GCNConv(in_channels, hidden_channels)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract only gene nodes' representations
        #gene_x = x[:num_genes]
        gene_x = x[num_genes:num_genes + num_pathways] # here only using the pathway features

        # Global pooling
        pooled = torch.mean(gene_x, dim=0).unsqueeze(0)

        # Final prediction
        out = self.lin(pooled)

        return out


class BipartiteAttentionHGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(BipartiteAttentionHGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Separate attention for gene and pathway features
        self.gene_attention = nn.Parameter(torch.ones(hidden_channels))
        self.pathway_attention = nn.Parameter(torch.ones(hidden_channels))

        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        # Apply GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Split features into gene and pathway features
        gene_features = x[:num_genes]  # Shape: [num_genes, hidden_channels]
        pathway_features = x[num_genes:num_genes + num_pathways]  # Shape: [num_pathways, hidden_channels]

        # Apply attention to gene and pathway features
        gene_weights = F.softmax(self.gene_attention, dim=0)
        pathway_weights = F.softmax(self.pathway_attention, dim=0)

        # Apply weighted mean to get a single vector for genes and pathways
        gene_summary = torch.sum(gene_features * gene_weights, dim=0)
        pathway_summary = torch.sum(pathway_features * pathway_weights, dim=0)

        # Combine gene and pathway information
        # Option 1: Simple addition
        combined = gene_summary + pathway_summary

        # Option 2: Concatenation followed by projection (alternative approach)
        # combined = torch.cat([gene_summary, pathway_summary])
        # self.projection = nn.Linear(2 * hidden_channels, hidden_channels)
        # combined = self.projection(combined)

        # Add batch dimension for classification
        pooled = combined.unsqueeze(0)

        # Final prediction
        out = self.lin(pooled)

        return out


class HierAttnBipartiteHGNN(nn.Module):
    """
    Gene–Pathway hypergraph GNN with
      • edge-level attention (gene→pathway)
      • pathway-level attention pooling
    """

    def __init__(self,
                 in_channels,        # gene expression dim
                 hidden_channels,            # embedding dim d
                 n_classes,
                 n_layers=2,
                 heads=4,
                 dropout=0.5):
        super().__init__()

        self.n_layers = n_layers

        # # 1-D gene expression → vector
        # self.gene_encoder = nn.Linear(in_channels, hidden)
        #
        # self.pathway_embed = nn.Parameter(torch.zeros(in_channels, hidden))

        # GCN layers for the bipartite representation
        self.conv1 = GCNConv(in_channels, hidden_channels)

        if n_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(n_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # pathway-level attention gate 𝑔(·)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        # final classifier
        self.lin = nn.Linear(hidden_channels, n_classes)
        self.dropout = dropout

    def forward(self, data, return_att=False):

        # gene_x = self.gene_encoder(data.x)                  # [N_g, d]
        # pathway_x = self.pathway_embed.expand(data.num_pathways, -1)

        x = data.x                 # [N_g, d]
        edge_index = data.edge_index

        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # split back
        gene_x, pathway_x = x[:data.num_genes], x[data.num_genes:]

        # Pathway-level attention
        path_attn_scores = self.gate_nn(pathway_x).squeeze(-1)  # [num_pathways]
        path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]

        # Create a graph-level embedding by weighting pathway features
        graph_emb = (path_weights.unsqueeze(-1) * pathway_x).sum(dim=0, keepdim=True)  # [1, hidden]

        out = self.lin(graph_emb)

        if return_att:
            return out, path_weights.detach(), None

        return out

class MLPBaseline(nn.Module):
    """Simple MLP baseline model for gene expression classification."""

    def __init__(self, input_size, hidden_size=512, num_classes=2, dropout_rate=0.1):
        """
        Args:
            input_size: Number of input features (genes)
            hidden_size: Size of hidden layers
            num_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(MLPBaseline, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x, H):
        # First layer
        x = x.view(1, -1)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        return x


# class SimpleHGNN(nn.Module):
#     def __init__(self, in_features, hidden_features, n_classes):
#         super().__init__()
#
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dropout = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(hidden_features, n_classes)
#
#     def forward(self, x, H):
#
#         x = x.view(1, -1)  # [1, num_genes]
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x

# class SimpleHGNN(nn.Module):
#     def __init__(self, in_features, hidden_features, n_classes):
#         super().__init__()
#
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dropout = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(hidden_features, n_classes)
#
#     def forward(self, x, H):
#
#         x = x.view(1, -1)  # [1, num_genes]
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#
#         # Hypergraph convolution
#         # Project x back to [num_genes, hidden_size] to match H
#         x = x.view(-1, x.shape[-1])  # [num_genes, hidden_size]
#
#         Dv_inv = torch.diag(1.0 / (torch.sum(H, dim=1) + 1e-6))
#         De_inv = torch.diag(1.0 / (torch.sum(H, dim=0) + 1e-6))
#         HT = H.T
#
#         x = Dv_inv @ H @ De_inv @ HT @ x  # [num_genes, hidden_size]
#
#         # Pool to get patient representation
#         pooled = x.mean(dim=0, keepdim=True)  # [1, hidden_size]
#
#         # Final classifier
#         out = self.fc2(pooled)  # [1, n_classes]
#
#         return out

#
# class SimpleHGNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
#         """
#         A simple Hypergraph Neural Network for gene expression data.
#
#         Args:
#             in_channels: Number of input features per node (typically 1 for gene expression)
#             hidden_channels: Size of hidden layers
#             out_channels: Number of output classes
#             num_layers: Number of hypergraph convolution layers
#             dropout: Dropout probability
#         """
#         super(SimpleHGNN, self).__init__()
#
#         self.num_layers = num_layers
#         self.dropout = dropout
#
#         # First hypergraph convolution layer
#         self.conv1 = HypergraphConv(in_channels, hidden_channels)
#
#         # Additional hypergraph convolution layers (if num_layers > 1)
#         if num_layers > 1:
#             self.convs = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
#
#         # Output layer
#         self.lin = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, data):
#         """
#         Forward pass through the network.
#
#         Args:
#             data: PyG Data object containing:
#                 - x: Node features [num_nodes, in_channels]
#                 - H: Incidence matrix [num_nodes, num_hyperedges]
#                 - num_genes: Number of gene nodes
#
#         Returns:
#             Logits for classification
#         """
#         x, H = data.x, data.H
#         num_genes = data.num_genes
#
#         indices = torch.nonzero(H).t().long()
#         values = H[indices[0], indices[1]]
#         H_sparse = FloatTensor(indices, values, H.size())
#
#         # First layer
#         x = self.conv1(x, H_sparse)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Additional layers (if any)
#         for i in range(self.num_layers - 1):
#             x = self.convs[i](x, H_sparse)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Extract only the gene nodes' representations
#         # (assuming the gene nodes are the first num_genes nodes)
#         gene_x = x[:num_genes]
#
#         # Global pooling to get a single vector per sample
#         # This combines all gene representations into one
#         pooled = torch.mean(gene_x, dim=0)
#
#         # Final prediction
#         out = self.lin(pooled)
#
#         return out


