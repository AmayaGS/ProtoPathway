"""
Multimodal Survival Baselines: SurvPath, MCAT, PIBD

Simplified implementations that capture the core ideas:
- SurvPath: Pathway tokens + patch tokens -> Multimodal Transformer
- MCAT: Genomic embeddings + patches -> Co-attention
- PIBD: Prototypical Information Bottleneck + Disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SurvPath (Jaume et al., CVPR 2024)
# =============================================================================

class SurvPath(nn.Module):
    """
    SurvPath: Pathway tokens + patch tokens with multimodal transformer.

    Key difference from ProtoPathway:
    - Uses fixed 300 hand-curated pathways
    - Each pathway encoded by independent MLP (no graph structure)
    - Dense multimodal attention between all tokens
    """

    def __init__(
            self,
            wsi_input_dim=768,
            num_pathways=300,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            num_classes=4,
            dropout=0.1,
    ):
        super().__init__()

        # Pathway encoding (MLP per pathway group)
        self.pathway_encoder = nn.Sequential(
            nn.Linear(num_pathways, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Patch projection
        self.patch_proj = nn.Linear(wsi_input_dim, hidden_dim)

        # Multimodal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable tokens
        self.pathway_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.wsi_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, wsi_features, pathway_features, **kwargs):
        """
        Args:
            wsi_features: [num_patches, wsi_input_dim]
            pathway_features: [num_pathways] - pathway expression values
        """
        # Encode pathways
        pathway_emb = self.pathway_encoder(pathway_features.unsqueeze(0))  # [1, hidden_dim]
        pathway_emb = pathway_emb.unsqueeze(1)  # [1, 1, hidden_dim]

        # Encode patches
        patch_emb = self.patch_proj(wsi_features).unsqueeze(0)  # [1, num_patches, hidden_dim]

        # Concatenate with learnable tokens
        tokens = torch.cat([
            self.pathway_token,
            pathway_emb,
            self.wsi_token,
            patch_emb
        ], dim=1)

        # Transformer
        out = self.transformer(tokens)

        # Pool from special tokens
        pathway_out = out[:, 0, :]
        wsi_out = out[:, 2, :]

        fused = torch.cat([pathway_out, wsi_out], dim=1)
        logits = self.classifier(fused)

        return {'logits': logits}


# =============================================================================
# MCAT (Chen et al., ICCV 2021)
# =============================================================================

class MCAT(nn.Module):
    """
    Multimodal Co-Attention Transformer.

    Key idea: Genomic embeddings attend to WSI patches via co-attention.
    """

    def __init__(
            self,
            wsi_input_dim=1024,
            omic_dim=80,
            hidden_dim=256,
            num_heads=4,
            num_classes=4,
            dropout=0.25,
    ):
        super().__init__()

        # WSI branch
        self.wsi_proj = nn.Sequential(
            nn.Linear(wsi_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention for WSI
        self.wsi_attn_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.wsi_attn_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Sigmoid())
        self.wsi_attn_W = nn.Linear(hidden_dim // 2, 1)

        # Omic branch
        self.omic_proj = nn.Sequential(
            nn.Linear(omic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Co-attention
        self.coattn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, wsi_features, omic_features, **kwargs):
        """
        Args:
            wsi_features: [num_patches, wsi_input_dim]
            omic_features: [omic_dim] - gene signature values
        """
        # WSI branch
        h_wsi = self.wsi_proj(wsi_features)

        A_V = self.wsi_attn_V(h_wsi)
        A_U = self.wsi_attn_U(h_wsi)
        A = self.wsi_attn_W(A_V * A_U)
        A = F.softmax(A, dim=0)
        wsi_pool = (A * h_wsi).sum(dim=0)

        # Omic branch
        h_omic = self.omic_proj(omic_features.unsqueeze(0))  # [1, hidden_dim]

        # Co-attention: omic queries WSI
        h_wsi_batch = h_wsi.unsqueeze(0)  # [1, num_patches, hidden_dim]
        h_omic_batch = h_omic.unsqueeze(0)  # [1, 1, hidden_dim]

        coattn_out, _ = self.coattn(h_omic_batch, h_wsi_batch, h_wsi_batch)
        coattn_out = coattn_out.squeeze()  # [hidden_dim]

        # Fusion
        fused = torch.cat([wsi_pool, coattn_out], dim=0)
        fused = self.fusion(fused.unsqueeze(0))

        logits = self.classifier(fused)

        return {'logits': logits}


# =============================================================================
# PIBD (Zhang et al., ICLR 2024)
# =============================================================================

class PIBD(nn.Module):
    """
    Prototypical Information Bottlenecking and Disentangling.

    Key ideas:
    - PIB: Learn prototypes for different risk levels to reduce intra-modal redundancy
    - PID: Disentangle modality-common and modality-specific information
    """

    def __init__(
            self,
            wsi_input_dim=768,
            pathway_dim=256,
            hidden_dim=256,
            num_prototypes=4,
            num_classes=4,
            dropout=0.1,
    ):
        super().__init__()

        # Projections
        self.wsi_proj = nn.Linear(wsi_input_dim, hidden_dim)
        self.pathway_proj = nn.Linear(pathway_dim, hidden_dim)

        # Prototypes for each modality
        self.wsi_prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        self.pathway_prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Disentanglement encoders
        self.common_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.specific_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, wsi_features, pathway_features, **kwargs):
        """
        Args:
            wsi_features: [num_patches, wsi_input_dim]
            pathway_features: [num_pathways, pathway_dim]
        """
        # Project
        h_wsi = self.wsi_proj(wsi_features)  # [N, D]
        h_pathway = self.pathway_proj(pathway_features)  # [P, D]

        # PIB: assign to prototypes
        wsi_dists = torch.cdist(h_wsi, self.wsi_prototypes)
        wsi_assign = F.softmax(-wsi_dists / self.temperature, dim=1)
        wsi_proto = torch.einsum('nk,nd->kd', wsi_assign, h_wsi)
        wsi_proto = wsi_proto / (wsi_assign.sum(0, keepdim=True).T + 1e-8)

        pathway_dists = torch.cdist(h_pathway, self.pathway_prototypes)
        pathway_assign = F.softmax(-pathway_dists / self.temperature, dim=1)
        pathway_proto = torch.einsum('nk,nd->kd', pathway_assign, h_pathway)
        pathway_proto = pathway_proto / (pathway_assign.sum(0, keepdim=True).T + 1e-8)

        # PID: disentangle
        wsi_common = self.common_encoder(wsi_proto)
        pathway_common = self.common_encoder(pathway_proto)
        common = (wsi_common + pathway_common) / 2

        wsi_specific = self.specific_encoder(wsi_proto)
        pathway_specific = self.specific_encoder(pathway_proto)

        # Pool and fuse
        common_pool = common.mean(dim=0)
        wsi_spec_pool = wsi_specific.mean(dim=0)
        pathway_spec_pool = pathway_specific.mean(dim=0)

        fused = torch.cat([common_pool, wsi_spec_pool, pathway_spec_pool], dim=0)
        logits = self.classifier(fused.unsqueeze(0))

        return {'logits': logits}