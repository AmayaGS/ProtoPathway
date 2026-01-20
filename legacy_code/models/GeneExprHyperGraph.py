import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv


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

class BipartiteGATHGNN(torch.nn.Module):
    """
    Bipartite graph representation of HGNN for gene expression data.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(BipartiteGATHGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers for the bipartite representation
        self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=False)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, concat=False))

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

class BipartiteGAT_MHSA(torch.nn.Module):
    """
    Bipartite graph representation of HGNN for gene expression data.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(BipartiteGAT_MHSA, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers for the bipartite representation
        self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=False)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, concat=False))

        ## Multi-head self-attention layer
        self.mhsa = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=1, batch_first=True)
        # pathway-level attention gate 𝑔(·)
        # self.gate_nn = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels // 2, 1)
        # )

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

        pathway_x = x[num_genes:num_genes + num_pathways] # here only using the pathway features

        # Global Linear Attention
        path_attn_scores = self.gate_nn(pathway_x)# [num_pathways]
        path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]

        # # Create a graph-level embedding by weighting pathway features
        graph_emb = (path_weights * pathway_x).sum(dim=0)  # [1, hidden]

        # Full self-attention
        # attn_out, attn_weights = self.mhsa(pathway_x, pathway_x, pathway_x)
        #
        # # attn_weights shape: [1, num_pathways, num_pathways]
        # pathway_importance = attn_weights.mean(dim=1)
        # #attn_out = attn_out
        # graph_emb = (pathway_importance.unsqueeze(-1) * attn_out).sum(dim=0)
        #
        # Mean pooling
        # graph_emb = torch.mean(pathway_x, dim=0)

        # Final prediction
        out = self.lin(graph_emb).unsqueeze(0)

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

        self.input_size = input_size

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, data):
        x = data.x

        x = x[:self.input_size]  # Select only the gene features

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