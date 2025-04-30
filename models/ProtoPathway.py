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

        # To store importance values
        self.pathway_importance = None
        self.gene_pathway_attention = None

    def forward(self, data, return_importance=False):
        x, edge_index = data.x, data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        attn_weights_list = []

        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        for i in range(self.num_layers - 1):
            if return_importance:
                x, (_, attn_weights) = self.convs[i](x, edge_index, return_attention_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = self.convs[i](x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pathway-level features
        pathway_x = x[num_genes:num_genes + num_pathways]

        # # Pathway-level attention
        path_attn_scores = self.gate_nn(pathway_x) # [num_pathways]
        path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]

        if return_importance:
            self.pathway_importance = path_weights.squeeze(-1).detach()

            final_attn = attn_weights_list[-1]

            self.gene_pathway_attention = self._process_gene_pathway_attention(edge_index, final_attn, num_genes, num_pathways)

        # # Create a graph-level embedding by weighting pathway features
        graph_emb = (path_weights * pathway_x).sum(dim=0)  # [hidden_dim]

        # Final prediction
        out = self.lin(graph_emb).unsqueeze(0) # [1, num_classes]

        return out


    def _process_gene_pathway_attention(self, edge_index, attn_weights, num_genes, num_pathways):

        gene_pathway_matrix = torch.zeros(num_genes, num_pathways, device=edge_index.device)

        for idx, (src, dst) in enumerate(edge_index.t()):
            # Only consider edges from genes to pathways
            if src < num_genes and dst >= num_genes and dst < num_genes + num_pathways:
                gene_idx = src.item()
                # Convert dst to pathway index
                pathway_idx = dst.item() - num_genes
                gene_pathway_matrix[gene_idx, pathway_idx] = attn_weights[idx].item()

        return gene_pathway_matrix


    def get_pathway_importance(self):
        return self.pathway_importance

    def get_gene_pathway_attention(self):
        return self.gene_pathway_attention

    def get_gene_importance(self):
        # Weight the gene importance by pathway importance and sum
        # This gives us a single importance score for each gene
        weighted_gene_importance = self.gene_pathway_attention @ self.pathway_importance
        return weighted_gene_importance

