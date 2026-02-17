import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class PathwayEmbeddingModel(torch.nn.Module):
    """
    Bipartite graph representation of HGNN for gene expression data.
    """
    def __init__(self, config, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, gene_names=None,
                 pathway_names=None):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.config = config

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

        self.gene_names = gene_names
        self.pathway_names = pathway_names

        # To store importance values
        self.gene_pathway_attention = None

    def forward(self, data, return_importance=False):
        x, edge_index = data.x, data.edge_index
        num_genes = data.num_genes
        num_pathways = data.num_pathways

        attn_weights_list = []

        # First layer
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        for i in range(self.num_layers - 1):
            if return_importance:
                x, (_, attn_weights) = self.convs[i](x, edge_index, return_attention_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = self.convs[i](x, edge_index)

            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pathway-level features
        pathway_x = x[num_genes:num_genes + num_pathways]

        if return_importance:
            final_attn = attn_weights_list[-1]
            self.gene_pathway_attention = self._process_gene_pathway_attention(edge_index, final_attn, num_genes, num_pathways)

        # # Pathway-level attention
        path_attn_scores = self.gate_nn(pathway_x) # [num_pathways]
        path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]
        # min max scaling
        # path_attn_scores = (path_attn_scores - path_attn_scores.min()) / (path_attn_scores.max() - path_attn_scores.min())
        # sigmoid scaling
        # path_attn_scores = torch.sigmoid(path_attn_scores)

        # # Create a graph-level embedding by weighting pathway features
        weighted_pathway = path_weights * pathway_x
        graph_emb = (weighted_pathway).sum(dim=0).unsqueeze(0)  # [hidden_dim]

        #pooled = torch.mean(pathway_x, dim=0).unsqueeze(0)

        # Final prediction
        out = self.lin(graph_emb).unsqueeze(0) # [1, num_classes]

        if self.config['execution']['mode'] == 'multimodal':
            return pathway_x, graph_emb
        else:
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


    def get_gene_pathway_attention(self):
        return self.gene_pathway_attention

    def get_gene_importance(self):
        # Weight the gene importance by pathway importance and sum
        # This gives us a single importance score for each gene
        weighted_gene_importance = self.gene_pathway_attention @ 1
        return weighted_gene_importance

    def set_names(self, gene_names, pathway_names):
        self.gene_names = gene_names
        self.pathway_names = pathway_names

    def get_top_genes(self, top_k=10):

        gene_imp = self.get_gene_importance()
        if gene_imp is None:
            raise ValueError("Gene importance not computed. Set return_importance=True during forward pass.")

        if self.gene_names is None:
            raise ValueError("Gene names not set. Use set_names() to set gene names.")

        gene_imp = gene_imp.cpu().numpy()
        top_indices = gene_imp.argsort()[-top_k:][::-1]

        top_genes = pd.DataFrame({
            'gene_name': [self.gene_names[i] for i in top_indices],
            'importance_score': gene_imp[top_indices]
        })

        return top_genes

    def get_top_genes_for_pathway(self, pathway_name=None, pathway_idx=None, top_k=10):

        if self.gene_pathway_attention is None:
            return None

        if self.gene_names is None:
            raise ValueError("Gene names not set. Use set_names() to set gene names.")

        if pathway_idx is None:
            if pathway_name is None:
                raise ValueError("Either pathway_name or pathway_idx must be provided")

        if self.pathway_names is None:
            raise ValueError("Pathway names have not been set. Use set_names() first.")

        try:
            pathway_idx = self.pathway_names.index(pathway_name)
        except ValueError:
            raise ValueError(f"Pathway {pathway_name} not found in pathway names.")

        # Get the attention scores for the specified pathway
        gene_pathway_imp = self.gene_pathway_attention[:, pathway_idx].cpu().numpy()

        top_indices = gene_pathway_imp.argsort()[-top_k:][::-1]

        top_genes = pd.DataFrame({
            'gene_name': [self.gene_names[i] for i in top_indices],
            'importance_score': gene_pathway_imp[top_indices]
        })

        return top_genes

    def generate_importance_report(self, top_genes=10, top_genes_per_pathway=5):

        if self.pathway_importance is None or self.gene_pathway_attention is None:
            raise ValueError(
                "Model has not calculated importance scores. Run forward with return_importance=True first.")

        report = {
            'top_genes': self.get_top_genes(top_k=top_genes),
            'genes_by_pathway': {}
        }

        return report

#
# class PathwayEmbeddingModel(torch.nn.Module):
#     """
#     Bipartite graph representation of HGNN for gene expression data.
#     """
#     def __init__(self, config, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, gene_names=None,
#                  pathway_names=None):
#         super().__init__()
#
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.config = config
#
#         # GAT layers for the bipartite representation
#         self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=False)
#
#         if num_layers > 1:
#             self.convs = nn.ModuleList()
#             for _ in range(num_layers - 1):
#                 self.convs.append(GATv2Conv(hidden_channels, hidden_channels, concat=False))
#
#         # pathway-level attention gate 𝑔(·)
#         self.gate_nn = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_channels // 2, 1)
#         )
#
#         # Output layer
#         self.lin = nn.Linear(hidden_channels, out_channels)
#
#         self.gene_names = gene_names
#         self.pathway_names = pathway_names
#
#         # To store importance values
#         self.pathway_importance = None
#         self.gene_pathway_attention = None
#
#     def forward(self, data, return_importance=False):
#         x, edge_index = data.x, data.edge_index
#         num_genes = data.num_genes
#         num_pathways = data.num_pathways
#
#         attn_weights_list = []
#
#         # First layer
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Additional layers
#         for i in range(self.num_layers - 1):
#             if return_importance:
#                 x, (_, attn_weights) = self.convs[i](x, edge_index, return_attention_weights=True)
#                 attn_weights_list.append(attn_weights)
#             else:
#                 x = self.convs[i](x, edge_index)
#
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#
#         # Pathway-level features
#         pathway_x = x[num_genes:num_genes + num_pathways]
#
#         # # Pathway-level attention
#         path_attn_scores = self.gate_nn(pathway_x) # [num_pathways]
#         path_weights = F.softmax(path_attn_scores, dim=0)  # [num_pathways]
#
#         if return_importance:
#             self.pathway_importance = path_weights.squeeze(-1).detach()
#
#             final_attn = attn_weights_list[-1]
#
#             self.gene_pathway_attention = self._process_gene_pathway_attention(edge_index, final_attn, num_genes, num_pathways)
#
#         # # Create a graph-level embedding by weighting pathway features
#         graph_emb = (path_weights * pathway_x).sum(dim=0).unsqueeze(0)  # [hidden_dim]
#         pooled = torch.mean(pathway_x, dim=0).unsqueeze(0)
#
#         # Final prediction
#         out = self.lin(graph_emb).unsqueeze(0) # [1, num_classes]
#
#         if self.config['execution']['mode'] == 'multimodal':
#             return pathway_x, pooled
#         else:
#             # Return only the output for single modality
#             return out
#
#
#     def _process_gene_pathway_attention(self, edge_index, attn_weights, num_genes, num_pathways):
#
#         gene_pathway_matrix = torch.zeros(num_genes, num_pathways, device=edge_index.device)
#
#         for idx, (src, dst) in enumerate(edge_index.t()):
#             # Only consider edges from genes to pathways
#             if src < num_genes and dst >= num_genes and dst < num_genes + num_pathways:
#                 gene_idx = src.item()
#                 # Convert dst to pathway index
#                 pathway_idx = dst.item() - num_genes
#                 gene_pathway_matrix[gene_idx, pathway_idx] = attn_weights[idx].item()
#
#         return gene_pathway_matrix
#
#
#     def get_pathway_importance(self):
#         return self.pathway_importance
#
#     def get_gene_pathway_attention(self):
#         return self.gene_pathway_attention
#
#     def get_gene_importance(self):
#         # Weight the gene importance by pathway importance and sum
#         # This gives us a single importance score for each gene
#         weighted_gene_importance = self.gene_pathway_attention @ self.pathway_importance
#         return weighted_gene_importance
#
#     def set_names(self, gene_names, pathway_names):
#         self.gene_names = gene_names
#         self.pathway_names = pathway_names
#
#     def get_top_pathways(self, top_k=10):
#         if self.pathway_importance is None:
#             raise ValueError("Pathway importance not computed. Set return_importance=True during forward pass.")
#
#         if self.pathway_names is None:
#             raise ValueError("Pathway names not set. Use set_names() to set pathway names.")
#
#         pathway_imp = self.pathway_importance.cpu().numpy()
#
#         top_indices = pathway_imp.argsort()[-top_k:][::-1]
#
#         top_pathways = pd.DataFrame({
#             'pathway_name': [self.pathway_names[i] for i in top_indices],
#             'importance_score': pathway_imp[top_indices]
#         })
#
#         return top_pathways
#
#     def get_top_genes(self, top_k=10):
#
#         gene_imp = self.get_gene_importance()
#         if gene_imp is None:
#             raise ValueError("Gene importance not computed. Set return_importance=True during forward pass.")
#
#         if self.gene_names is None:
#             raise ValueError("Gene names not set. Use set_names() to set gene names.")
#
#         gene_imp = gene_imp.cpu().numpy()
#         top_indices = gene_imp.argsort()[-top_k:][::-1]
#
#         top_genes = pd.DataFrame({
#             'gene_name': [self.gene_names[i] for i in top_indices],
#             'importance_score': gene_imp[top_indices]
#         })
#
#         return top_genes
#
#     def get_top_genes_for_pathway(self, pathway_name=None, pathway_idx=None, top_k=10):
#
#         if self.gene_pathway_attention is None:
#             return None
#
#         if self.gene_names is None:
#             raise ValueError("Gene names not set. Use set_names() to set gene names.")
#
#         if pathway_idx is None:
#             if pathway_name is None:
#                 raise ValueError("Either pathway_name or pathway_idx must be provided")
#
#         if self.pathway_names is None:
#             raise ValueError("Pathway names have not been set. Use set_names() first.")
#
#         try:
#             pathway_idx = self.pathway_names.index(pathway_name)
#         except ValueError:
#             raise ValueError(f"Pathway {pathway_name} not found in pathway names.")
#
#         # Get the attention scores for the specified pathway
#         gene_pathway_imp = self.gene_pathway_attention[:, pathway_idx].cpu().numpy()
#
#         top_indices = gene_pathway_imp.argsort()[-top_k:][::-1]
#
#         top_genes = pd.DataFrame({
#             'gene_name': [self.gene_names[i] for i in top_indices],
#             'importance_score': gene_pathway_imp[top_indices]
#         })
#
#         return top_genes
#
#     def generate_importance_report(self, top_pathways=10, top_genes=10, top_genes_per_pathway=10):
#
#         if self.pathway_importance is None or self.gene_pathway_attention is None:
#             raise ValueError(
#                 "Model has not calculated importance scores. Run forward with return_importance=True first.")
#
#         report = {
#             'top_pathways': self.get_top_pathways(top_k=top_pathways),
#             'top_genes': self.get_top_genes(top_k=top_genes),
#             'genes_by_pathway': {}
#         }
#
#         # Get top genes for each of the top pathways
#         top_pathway_df = report['top_pathways']
#         for idx, row in top_pathway_df.iterrows():
#             pathway_name = row['pathway_name']
#             report['genes_by_pathway'][pathway_name] = self.get_top_genes_for_pathway(
#                 pathway_name=pathway_name,
#                 top_k=top_genes_per_pathway
#             )
#
#         return report
#
#
