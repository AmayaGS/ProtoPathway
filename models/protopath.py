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

from models.protopath_components.gene_encoder import PathwayEmbeddingModel
from models.protopath_components.wsi_encoder import PrototypeMIL
from models.protopath_components.fusion import get_fusion_module


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
        gene_hidden_dim=128,
        gene_num_layers=3,
        gene_dropout=0.25,
        gene_num_heads=4,
        gene_enabled=True,
        # WSI encoder params
        wsi_input_dim=1536,
        wsi_hidden_dim=128,
        wsi_num_prototypes=16,
        wsi_tau=0.1,
        wsi_centroids=None,
        wsi_enabled=True,
        # Fusion params
        fusion_type='cross_attention',
        fusion_num_heads=2,
        fusion_dropout=0.25,
        fusion_output_dim=64,
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
            fusion_output_dim: Dimensionality of fused embedding (input to classifier)
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
                output_dim=fusion_output_dim,
                num_heads=fusion_num_heads,
                dropout=fusion_dropout
            )
            self.projection = nn.Identity()
            logging.info(
                f"Fusion: {fusion_type}, output_dim={fusion_output_dim}, "
                f"classifier {fusion_output_dim}->{num_classes}"
            )
        else:
            # Unimodal - no fusion needed
            self.fusion = None
            unimodal_dim = gene_hidden_dim if gene_enabled else wsi_hidden_dim
            self.projection = nn.Sequential(
                nn.Linear(unimodal_dim, fusion_output_dim),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
            )
            logging.info("Unimodal mode - no fusion")

        self.classifier = nn.Linear(fusion_output_dim, num_classes)

        # Storage for visualization outputs
        self.last_attention_weights = None

    def forward(self, data, return_attention=False, return_embeddings=False):
        """
        Forward pass through ProtoPathway.

        Args:
            data: PyG Data object containing:
                - x: Gene node features [num_genes + num_pathways, 1]
                - edge_index: Bipartite graph edges
                - num_genes, num_pathways: Graph structure info
                - wsi_features: WSI patch features [num_patches, num_features]
            return_attention: Whether to compute and store attention weights
            return_embeddings: Whether to return per-modality embeddings

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
            embedding = self.projection(embedding)

        # Classification
        logits = self.classifier(embedding.unsqueeze(0))  # [1, num_classes]

        if return_embeddings:
            return logits, pathway_mean, wsi_embedding

        return logits

    def get_attention_outputs(self):
        """
        Get attention weights from the last forward pass.

        Returns dict with:
        - 'gene_pathway_attention': Gene-pathway attention from GATv2
        - 'pathway_importance': Pathway gate weights
        - 'patch_assignments': Patch-prototype assignments
        - 'fusion_gate_weights': Prototype-pathway gate weights (if available)
        - 'cross_modal_attention': Prototype-pathway cross attention (if applicable)
        """
        outputs = {}

        if self.gene_enabled and self.gene_encoder.gene_pathway_attention is not None:
            outputs['gene_pathway_attention'] = self.gene_encoder.gene_pathway_attention
            outputs['pathway_importance'] = self.gene_encoder.pathway_importance

        if self.wsi_enabled:
            outputs['patch_assignments'] = self.wsi_encoder.get_prototype_assignments()

            # All fusion modules expose proto_pathway_gate_weights (None if N/A)
            if self.fusion is not None and self.fusion.proto_pathway_gate_weights is not None:
                outputs['fusion_gate_weights'] = self.fusion.proto_pathway_gate_weights

        if self.last_attention_weights is not None:
            outputs['cross_modal_attention'] = self.last_attention_weights

        return outputs