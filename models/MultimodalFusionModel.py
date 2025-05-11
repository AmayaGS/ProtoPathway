# models/MultimodalFusionModel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ProtoPathway import PathwayEmbeddingModel
from models.Prototype import ProtoMIL_V0, ProtoMIL_V1


class ProtoPathwayFusion(torch.nn.Module):
    """
    Multimodal model for gene expression and WSI data using ProtoPathway and ProtoMIL.
    """
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Initialize the gene expression model
        self.ge_model = PathwayEmbeddingModel(config, in_channels=config['ge_training']['input_dim'],
                                              hidden_channels=config['ge_training']['hidden_dim'],
                                              out_channels=config['n_classes'],
                                              num_layers=config['ge_training']['num_layers'],
                                              dropout=config['ge_training']['dropout_rate']).to(device)

        # Initialize the WSI model
        self.wsi_model = ProtoMIL_V1(config, input_dim=config['wsi_training']['input_dim'],
                                     num_prototypes=config['wsi_training']['num_prototypes'],
                                     tau=config['wsi_training']['tau'], num_classes=config['n_classes']).to(device)

        # Initialize MHSA cross-attention between pathway embeddings and prototypes
        self.proto_pathway_attention = nn.MultiheadAttention(
                                        embed_dim=config['mm_training']['input_dim'],
                                        num_heads=config['mm_training']['attention_heads'],
                                        dropout=config['mm_training']['dropout_rate'],
                                        batch_first=True
                                        ).to(device) # change this to the mm_training after setup in config

        # Initialize the final classifier
        self.classifier = nn.Linear(config['mm_training']['input_dim'], config['n_classes']).to(device)

    def forward(self, ge_data, wsi_data):
        """
        Forward pass through the model.
        :param ge_data: Gene expression data
        :param wsi_data: WSI data
        :return: Model predictions
        """
        # Get gene expression embeddings
        pathway_emb, pathway_mean = self.ge_model(ge_data)

        # Get prototype hist and prototype tokens
        proto_hist, proto_tokens = self.wsi_model(wsi_data)

        # Cross-attention between prototypes and gene expression embeddings
        attended_proto, attention_weights = self.proto_pathway_attention(query=proto_tokens,
                                                                        key=pathway_emb,
                                                                        value=pathway_emb,
                                                                        need_weights=True,
                                                                        average_attn_weights=True
                                                                         )

        # Classifier on the attention output
        logits = self.classifier(attended_proto.squeeze(1))

        return logits
