"""
ProtoPathway: Unified Multimodal Model for Survival Prediction.

Combines:
- Gene expression encoding via bipartite GATv2 (PathwayEmbeddingModel)
- WSI encoding via prototype-based MIL (PrototypeMIL)
- Flexible fusion mechanisms (concat, cross-attention, bilinear, gated)

Supports ablation studies by enabling/disabling branches.
"""

import logging
import torch.nn as nn

from models.components.gene_encoder import PathwayEmbeddingModel
from models.components.wsi_encoder import PrototypeMIL
from models.components.fusion import get_fusion_module


class ProtoPathway(nn.Module):
    """
    Unified ProtoPathway model for multimodal survival/classification prediction.

    Architecture:
    1. Gene encoder: Bipartite GATv2 over pathway-gene graph
    2. WSI encoder: Prototype-based MIL
    3. Fusion: Combines modality embeddings (multiple strategies available)
    4. Classifier: Predicts survival bins or class labels

    Supports ablation by disabling branches (gene_enabled, wsi_enabled).
    """

    def __init__(
        self,
        num_classes,
        # Gene encoder params
        gene_hidden_dim=256,
        gene_num_layers=2,
        gene_dropout=0.3,
        gene_num_heads=1,
        gene_enabled=True,
        # WSI encoder params
        wsi_input_dim=1536,  # UNI2-h default
        wsi_hidden_dim=256,
        wsi_num_prototypes=64,
        wsi_tau=10.0,
        wsi_centroids=None,
        wsi_enabled=True,
        # Fusion params
        fusion_type='cross_attention',
        fusion_num_heads=4,
        fusion_dropout=0.3
    ):
        """
        Initialize ProtoPathway.

        Args:
            num_classes: Number of output classes (survival bins or classification classes)
            gene_*: Gene encoder parameters
            gene_enabled: Whether to use gene expression branch
            wsi_*: WSI encoder parameters
            wsi_centroids: Optional pre-computed centroids for prototype init
            wsi_enabled: Whether to use WSI branch
            fusion_*: Fusion mechanism parameters
        """
        super().__init__()

        self.num_classes = num_classes
        self.gene_enabled = gene_enabled
        self.wsi_enabled = wsi_enabled
        self.fusion_type = fusion_type

        # Validate at least one branch is enabled
        if not gene_enabled and not wsi_enabled:
            raise ValueError("At least one branch (gene or WSI) must be enabled")

        # Determine hidden dimensions
        self.gene_hidden_dim = gene_hidden_dim if gene_enabled else 0
        self.wsi_hidden_dim = wsi_hidden_dim if wsi_enabled else 0

        # Gene encoder
        if gene_enabled:
            self.gene_encoder = PathwayEmbeddingModel(
                hidden_dim=gene_hidden_dim,
                num_layers=gene_num_layers,
                dropout=gene_dropout,
                num_heads=gene_num_heads
            )
            logging.info(f"Gene encoder: {gene_num_layers} GATv2 layers, dim={gene_hidden_dim}")
        else:
            self.gene_encoder = None
            logging.info("Gene encoder: DISABLED")

        # WSI encoder
        if wsi_enabled:
            self.wsi_encoder = PrototypeMIL(
                input_dim=wsi_input_dim,
                hidden_dim=wsi_hidden_dim,
                num_prototypes=wsi_num_prototypes,
                tau=wsi_tau,
                init_centroids=wsi_centroids
            )
            logging.info(f"WSI encoder: {wsi_num_prototypes} prototypes, dim={wsi_hidden_dim}")
        else:
            self.wsi_encoder = None
            logging.info("WSI encoder: DISABLED")

        # Fusion and classifier
        if gene_enabled and wsi_enabled:
            # Multimodal fusion
            hidden_dim = gene_hidden_dim  # Assumes same dim for both
            self.fusion = get_fusion_module(
                fusion_type=fusion_type,
                hidden_dim=hidden_dim,
                num_heads=fusion_num_heads,
                dropout=fusion_dropout
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
            logging.info(f"Fusion: {fusion_type}, classifier dim={hidden_dim}")
        else:
            # Unimodal - no fusion needed
            self.fusion = None
            if gene_enabled:
                self.classifier = nn.Linear(gene_hidden_dim, num_classes)
            else:
                self.classifier = nn.Linear(wsi_hidden_dim, num_classes)
            logging.info("Unimodal mode - no fusion")

        # Storage for visualization outputs
        self.last_attention_weights = None

    def forward(self, data, return_attention=False):
        """
        Forward pass through ProtoPathway.

        Args:
            data: PyG Data object containing:
                - x: Gene node features [num_genes + num_pathways, 1]
                - edge_index: Bipartite graph edges
                - num_genes, num_pathways: Graph structure info
                - wsi_features: WSI patch features [num_patches, wsi_input_dim]
            return_attention: Whether to compute and store attention weights

        Returns:
            logits: [1, num_classes] prediction logits
        """
        pathway_mean = None
        wsi_embedding = None
        pathway_embeddings = None
        proto_tokens = None

        # Gene encoder forward
        if self.gene_enabled:
            pathway_embeddings, pathway_mean = self.gene_encoder(
                data, return_attention=return_attention
            )

        # WSI encoder forward
        if self.wsi_enabled:
            wsi_features = data.wsi_features
            wsi_embedding, proto_tokens = self.wsi_encoder(
                wsi_features, return_assignments=return_attention
            )

        # Fusion
        if self.gene_enabled and self.wsi_enabled:
            fused, attn_weights = self.fusion(
                                            pathway_mean=pathway_mean,
                                            wsi_embedding=wsi_embedding,
                                            pathway_embeddings=pathway_embeddings,
                                            proto_tokens=proto_tokens
                                            )
            embedding = fused

            if return_attention:
                self.last_attention_weights = attn_weights
        else:
            # Unimodal
            embedding = pathway_mean if self.gene_enabled else wsi_embedding

        # Classification
        logits = self.classifier(embedding.unsqueeze(0))  # [1, num_classes]

        return logits

    def get_attention_outputs(self):
        """
        Get attention weights from the last forward pass.

        Returns dict with:
        - 'gene_pathway_attention': Gene-pathway attention from GATv2
        - 'pathway_importance': Pathway gate weights
        - 'patch_assignments': Patch-prototype assignments
        - 'cross_modal_attention': Prototype-pathway cross attention (if applicable)
        """
        outputs = {}

        if self.gene_enabled and self.gene_encoder.gene_pathway_attention is not None:
            outputs['gene_pathway_attention'] = self.gene_encoder.gene_pathway_attention
            outputs['pathway_importance'] = self.gene_encoder.pathway_importance

        if self.wsi_enabled:
            outputs['patch_assignments'] = self.wsi_encoder.get_prototype_assignments()
            outputs['prototype_gate_weights'] = self.wsi_encoder.get_gate_weights()

        if self.last_attention_weights is not None:
            outputs['cross_modal_attention'] = self.last_attention_weights

        return outputs


def build_protopath(cfg, wsi_centroids=None):
    """
    Build ProtoPathway model from config.

    Args:
        cfg: OmegaConf config object
        wsi_centroids: Optional pre-computed centroids tensor

    Returns:
        ProtoPathway model instance
    """
    # Determine number of classes
    if cfg.task == 'survival':
        num_classes = cfg.survival.num_bins
    else:
        num_classes = cfg.classification.num_classes

    model_cfg = cfg.model

    model = ProtoPathway(
        num_classes=num_classes,
        # Gene encoder
        gene_hidden_dim=model_cfg.gene_encoder.hidden_dim,
        gene_num_layers=model_cfg.gene_encoder.num_layers,
        gene_dropout=model_cfg.gene_encoder.dropout,
        gene_num_heads=model_cfg.gene_encoder.num_heads,
        gene_enabled=model_cfg.branches.gene,
        # WSI encoder
        wsi_input_dim=1536,  # UNI2-h
        wsi_hidden_dim=model_cfg.wsi_encoder.hidden_dim,
        wsi_num_prototypes=model_cfg.wsi_encoder.num_prototypes,
        wsi_tau=model_cfg.wsi_encoder.tau,
        wsi_centroids=wsi_centroids,
        wsi_enabled=model_cfg.branches.wsi,
        # Fusion
        fusion_type=model_cfg.fusion.type,
        fusion_num_heads=model_cfg.fusion.num_heads,
        fusion_dropout=model_cfg.fusion.dropout
    )

    return model