import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class PathwayEmbeddingModel(torch.nn.Module):
    """
    Bipartite graph representation of HGNN for gene expression data.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GAT layers for the bipartite representation
        self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=False)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, concat=False))

        # pathway-level attention gate 𝑔(·)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

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

        # Pathway-level features
        pathway_x = x[num_genes:num_genes + num_pathways]

        # # Pathway-level attention
        path_attn_scores = self.gate_nn(pathway_x) # [num_pathways]
        path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]

        # # Create a graph-level embedding by weighting pathway features
        graph_emb = (path_weights * pathway_x).sum(dim=0)  # [hidden_dim]

        # Final prediction
        out = self.lin(graph_emb).unsqueeze(0) # [1, num_classes]

        return out