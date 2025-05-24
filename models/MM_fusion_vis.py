# models/MultimodalFusionModel.py

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv



class ProtoPathwayFusion(torch.nn.Module):
    """
    Multimodal model for gene expression and WSI data using ProtoPathway and ProtoMIL.
    """
    def __init__(self, config, centroids, device):

        super().__init__()

        self.config = config
        self.device = device
        self.is_survival = config['execution'].get('task', 'classification') == 'survival'
        self.is_visualise = config['execution']['visualise']

        if self.is_survival:
            n_classes = config['survival']['survival_bins']
        else:
            n_classes = config['n_classes']


        # Initialize the gene expression model
        self.ge_model = PathwayEmbeddingModel(config, in_channels=config['ge_training']['input_dim'],
                                              hidden_channels=config['ge_training']['hidden_dim'],
                                              out_channels=n_classes,
                                              num_layers=config['ge_training']['num_layers'],
                                              dropout=config['ge_training']['dropout_rate']).to(device)

        # Initialize the WSI model
        self.wsi_model = ProtoMIL_V1(config, input_dim=config['wsi_training']['input_dim'],
                                     embedding_dim=config['wsi_training']['hidden_dim'],
                                     num_prototypes=config['wsi_training']['num_prototypes'],
                                     tau=config['wsi_training']['tau'],
                                     num_classes=n_classes,
                                     init_centroids=centroids).to(device)

        # Initialize MHSA cross-attention between pathway embeddings and prototypes
        self.proto_pathway_attention = nn.MultiheadAttention(
                                        embed_dim=config['mm_training']['hidden_dim'],
                                        num_heads=config['mm_training']['attention_heads'],
                                        dropout=config['mm_training']['dropout_rate'],
                                        batch_first=True
                                        ).to(device) # change this to the mm_training after setup in config

        # Initialize the final classifier
        self.classifier = nn.Linear(config['mm_training']['hidden_dim'] * 3, n_classes).to(device)


    def forward(self, ge_data, wsi_data):
        """
        Forward pass through the model.
        :param ge_data: Gene expression data
        :param wsi_data: WSI data
        :return: Model predictions
        """

        patient_id = ge_data['patient_id']
        attention_dict = {}

        # Get gene expression embeddings
        pathway_emb, pathway_mean, gene_pathway_attention = self.ge_model(ge_data)
        pathway_emb = pathway_emb.unsqueeze(0)  # Add a batch dimension for attention

        # Get prototype hist and prototype tokens
        proto_hist, proto_tokens, prototype_assignments, patch_similarities = self.wsi_model(wsi_data)

        # Cross-attention between prototypes and gene expression embeddings
        attended_proto, attention_weights = self.proto_pathway_attention(query=proto_tokens,
                                                                        key=pathway_emb,
                                                                        value=pathway_emb,
                                                                        need_weights=True,
                                                                        average_attn_weights=True
                                                                         )

        proto_path_mean = attended_proto.mean(dim=1)
        # Concatenate the mean pooled pathway embeddings and the attended prototypes
        combined_features = torch.cat((pathway_mean, proto_path_mean, proto_hist), dim=1)

        # Classifier on the attention output
        logits = self.classifier(combined_features)
        attention_dict = {
                        'gene_pathway_attn': gene_pathway_attention,
                        'prototype_assignments': prototype_assignments,
                        'patch_similarities': patch_similarities,
                        'cross_modal_attn': attention_weights
                        }

        return logits, attention_dict

class ProtoMIL_V1(nn.Module):
    """
    Smallest working prototype-MIL model.
    """
    def __init__(self, config, input_dim: int, embedding_dim, num_prototypes: int = 64, tau: float = 10.0,
                 num_classes: int = 2, init_centroids: torch.Tensor | None = None):

        super().__init__()

        self.config = config
        self.is_visualise = config['execution']['visualise']

        if init_centroids is None:
            print("Randomly initializing centroids")
            self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))
        else:
            print("Using pre-computed centroids for initialization")
            self.proto = nn.Parameter(init_centroids)          # k-means seeds

        # Add dimension reduction layer
        self.dim_reducer = nn.Linear(input_dim, embedding_dim)

        # (b) non-negative gates (soft-plus ensures ≥0)
        self.logit_g = nn.Parameter(torch.zeros(num_prototypes))

        self.tau = tau                                         # softmax temp
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.prototype_assignments = None
        self.patch_similarities = None


    def forward(self, x):
        """
        x  – patch-level embeddings for a batch of slides
             shape: [B_slide , P_patch , D]    (NOT [B, D])
        """
        x = x.unsqueeze(0)  # [B, P, D]  (batch size = 1)
        B, P, D = x.shape
        N = self.proto.shape[0]

        # 1) Reduce dimensions of both prototypes and input
        p_reduced = self.dim_reducer(self.proto)  # [N, embedding_dim]

        # Reshape input for batch processing through linear layer
        x_flat = x.view(-1, D)  # [B*P, D]
        x_reduced = self.dim_reducer(x_flat)  # [B*P, embedding_dim]
        x_reduced = x_reduced.view(B, P, -1)  # [B, P, embedding_dim]

        # 2) L2-normalise & cosine similarity
        p = F.normalize(p_reduced, dim=1)  # [N, D]
        x_n = F.normalize(x_reduced, dim=2)  # [B, P, D]
        sim = torch.einsum("bpd,nd->bpn", x_n, p)  # [B, P, N]

        # 3) soft assignment
        alpha = F.softmax(self.tau * sim, dim=2)  # [B, P, N]
        gates = F.softplus(self.logit_g)  # [N]
        alpha_g = alpha * gates  # broadcast to [B, P, N]

        # 4) prototype-wise pooling  →  tokens
        #    µ_{b,n} = Σ_p α̃_{bpn}·x_{bp}  /  Σ_p α̃_{bpn}
        numer = torch.einsum("bpn,bpd->bnd", alpha_g, x_reduced)  # [B, N, D]
        denom = alpha_g.sum(dim=1, keepdim=False).clamp(min=1e-6)  # [B, N]
        proto_tok = numer / denom.unsqueeze(2)  # [B, N, D]

        # 5) slide-level representation for the unimodal baseline
        #    (simple mean over prototype tokens)
        bag_repr = proto_tok.mean(dim=1)  # [B, D]
        logits = self.classifier(bag_repr)  # [B, C]

        if self.is_visualise:
            self.prototype_assignments = alpha
            self.patch_similarities = sim

        if self.config['execution']['mode'] == 'multimodal':
            return bag_repr, proto_tok, self.prototype_assignments, self.patch_similarities
        else:
            return logits, sim


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
        self.is_visualise = config['execution']['visualise']

        # GAT layers for the bipartite representation
        self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=False)

        if num_layers > 1:
            self.convs = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, concat=False))

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
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Additional layers
        for i in range(self.num_layers - 1):
            if self.is_visualise:
                x, (_, attn_weights) = self.convs[i](x, edge_index, return_attention_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = self.convs[i](x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pathway-level features
        pathway_x = x[num_genes:num_genes + num_pathways]

        if self.is_visualise:
            final_attn = attn_weights_list[-1]
            self.gene_pathway_attention = self._process_gene_pathway_attention(edge_index, final_attn, num_genes, num_pathways)

        # # Create a graph-level embedding by weighting pathway features
        pooled = torch.mean(pathway_x, dim=0).unsqueeze(0)

        # Final prediction
        out = self.lin(pooled).unsqueeze(0) # [1, num_classes]

        if self.config['execution']['mode'] == 'multimodal':
            return pathway_x, pooled, self.gene_pathway_attention
        else:
            # Return only the output for single modality
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

