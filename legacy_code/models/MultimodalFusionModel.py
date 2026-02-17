# models/MultimodalFusionModel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy_code.models.ProtoPathway import PathwayEmbeddingModel
from legacy_code.models.Prototype import ProtoMIL_V1


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

        # self.proto_pathway_attention = SimpleCrossAttention(
        #     embed_dim=config['mm_training']['hidden_dim'],
        #     dropout=config['mm_training']['dropout_rate']
        # )

        # Initialize the final classifier
        self.classifier = nn.Linear(config['mm_training']['hidden_dim'] * 3, n_classes).to(device)

    def forward(self, ge_data, wsi_data):
        """
        Forward pass through the model.
        :param ge_data: Gene expression data
        :param wsi_data: WSI data
        :return: Model predictions
        """
        # Get gene expression embeddings
        pathway_emb, pathway_mean = self.ge_model(ge_data)
        pathway_emb = pathway_emb.unsqueeze(0)  # Add a batch dimension for attention

        # Get prototype hist and prototype tokens
        proto_hist, proto_tokens = self.wsi_model(wsi_data)

        # # Cross-attention between prototypes and gene expression embeddings
        # attended_proto, attention_weights = self.proto_pathway_attention(query=proto_tokens,
        #                                                                 key=pathway_emb,
        #                                                                 value=pathway_emb,
        #                                                                 need_weights=True,
        #                                                                 average_attn_weights=True
        #                                                                  )

        raw_attention = torch.matmul(proto_tokens, pathway_emb.transpose(-2, -1))
        # attention_weights = raw_attention / raw_attention.sum(dim=-1, keepdim=True)  # L1 normalize
        attention_weights = (raw_attention - raw_attention.min()) / (raw_attention.max() - raw_attention.min())
        attended_proto = torch.matmul(attention_weights, pathway_emb)

        # attended_proto, attention_weights = self.proto_pathway_attention(
        #     query=proto_tokens,
        #     key=pathway_emb,
        #     value=pathway_emb
        # )

        # attention_scores = torch.matmul(proto_tokens, pathway_emb.transpose(-2, -1))
        # # attention_scores = torch.sigmoid(attention_scores)  # Apply sigmoid to get attention weights
        # attended_proto = torch.matmul(attention_scores, pathway_emb)

        proto_path_mean = attended_proto.mean(dim=1)
        # Concatenate the mean pooled pathway embeddings and the attended prototypes
        combined_features = torch.cat((pathway_mean, proto_path_mean, proto_hist), dim=1)

        # Classifier on the attention output
        logits = self.classifier(combined_features)

        return logits


class SimpleCrossAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = embed_dim ** -0.5

    def forward(self, query, key, value):
        B, N_q, D = query.shape
        B, N_kv, D = key.shape

        Q = self.query_proj(query)  # [B, N_q, D]
        K = self.key_proj(key)  # [B, N_kv, D]
        V = self.value_proj(value)  # [B, N_kv, D]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, N_q, N_kv]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(scores, V)  # [B, N_q, D]

        return attended, scores
