"""
Fusion Mechanisms for Multimodal Integration.

Provides multiple fusion strategies:
- Concatenation
- Cross-attention (pathway <-> prototype)
- Simple cross-attention
- Bilinear fusion
- Gated fusion

All modules output [output_dim] and expose a `proto_pathway_gate_weights`
attribute (None when not applicable) for a uniform attention interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.

    Concatenates pathway and prototype embeddings, then projects.
    """

    def __init__(self, gene_dim, wsi_dim, output_dim, dropout=0.3):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(gene_dim + wsi_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Uniform interface
        self.proto_pathway_gate_weights = None

    def forward(self, pathway_mean, wsi_embedding, pathway_embeddings=None, proto_tokens=None):
        """
        Args:
            pathway_mean: [hidden_dim] from gene encoder
            wsi_embedding: [hidden_dim] from WSI encoder
            pathway_embeddings: Not used (for interface consistency)
            proto_tokens: Not used (for interface consistency)

        Returns:
            fused: [output_dim] fused embedding
            attention_weights: None (no attention in concat)
        """
        combined = torch.cat([pathway_mean, wsi_embedding], dim=-1)
        fused = self.projection(combined)
        return fused, None


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between pathway embeddings and prototype tokens.

    Prototypes attend to pathways to create pathway-informed prototype representations.
    Final embedding combines: pathway_mean, attended_proto_mean, wsi_embedding
    """

    def __init__(self, hidden_dim, output_dim, num_heads=4, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Multi-head cross attention: prototypes (query) attend to pathways (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate_pp = nn.Linear(hidden_dim, 1)

        self.norm_gene = nn.LayerNorm(hidden_dim)
        self.norm_wsi = nn.LayerNorm(hidden_dim)
        self.norm_pp = nn.LayerNorm(hidden_dim)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proto_pathway_gate_weights = None

    def forward(self, pathway_mean, wsi_embedding, pathway_embeddings, proto_tokens):
        """
        Args:
            pathway_mean: [hidden_dim] from gene encoder
            wsi_embedding: [hidden_dim] from WSI encoder
            pathway_embeddings: [num_pathways, hidden_dim] individual pathway embeddings
            proto_tokens: [num_prototypes, hidden_dim] prototype embeddings

        Returns:
            fused: [output_dim] fused embedding
            attention_weights: [num_prototypes, num_pathways] cross-modal attention
        """
        pathways = pathway_embeddings.unsqueeze(0)   # [1, num_pathways, hidden_dim]
        prototypes = proto_tokens.unsqueeze(0)        # [1, num_prototypes, hidden_dim]

        # Cross attention: prototypes query pathways
        attended_proto, attn_weights = self.cross_attention(
            query=prototypes,
            key=pathways,
            value=pathways,
            need_weights=True,
            average_attn_weights=True
        )

        gates_proto = self.gate_pp(attended_proto)          # [1, N, 1]
        gates_proto = F.softmax(gates_proto, dim=1)
        weighted_proto = (gates_proto * attended_proto).sum(dim=1)  # [1, hidden_dim]

        self.proto_pathway_gate_weights = gates_proto.squeeze(0).squeeze(-1).detach()

        # Combine all three embeddings
        combined = torch.cat([
            self.norm_gene(pathway_mean),
            self.norm_pp(weighted_proto.squeeze(0)),
            self.norm_wsi(wsi_embedding)
        ], dim=-1)

        fused = self.projection(combined)

        return fused, attn_weights.squeeze(0)


class SimpleCrossAttention(nn.Module):
    """
    Simplified cross-attention without multi-head complexity.

    Uses learned query/key/value projections with scaled dot-product attention.
    """

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Uniform interface
        self.proto_pathway_gate_weights = None

    def forward(self, pathway_mean, wsi_embedding, pathway_embeddings, proto_tokens):
        """
        Args:
            pathway_mean: [hidden_dim] from gene encoder
            wsi_embedding: [hidden_dim] from WSI encoder
            pathway_embeddings: [num_pathways, hidden_dim]
            proto_tokens: [num_prototypes, hidden_dim]

        Returns:
            fused: [output_dim] fused embedding
            attention_weights: [num_prototypes, num_pathways]
        """
        Q = self.query_proj(proto_tokens)       # [N_proto, hidden_dim]
        K = self.key_proj(pathway_embeddings)    # [N_pathway, hidden_dim]
        V = self.value_proj(pathway_embeddings)  # [N_pathway, hidden_dim]

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # [N_proto, hidden_dim]
        attended_mean = attended.mean(dim=0)       # [hidden_dim]

        combined = torch.cat([pathway_mean, attended_mean, wsi_embedding], dim=-1)
        fused = self.projection(combined)

        return fused, attn_weights


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for capturing interactions between modalities.

    Uses low-rank factorization to reduce parameters.
    """

    def __init__(self, gene_dim, wsi_dim, output_dim, rank=64, dropout=0.3):
        super().__init__()

        self.rank = rank

        # Low-rank factorization: W = U @ V^T
        self.U = nn.Linear(gene_dim, rank, bias=False)
        self.V = nn.Linear(wsi_dim, rank, bias=False)

        # Combine bilinear term with original embeddings
        self.projection = nn.Sequential(
            nn.Linear(rank + gene_dim + wsi_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Uniform interface
        self.proto_pathway_gate_weights = None

    def forward(self, pathway_mean, wsi_embedding, pathway_embeddings=None, proto_tokens=None):
        """
        Args:
            pathway_mean: [hidden_dim] from gene encoder
            wsi_embedding: [hidden_dim] from WSI encoder

        Returns:
            fused: [output_dim] fused embedding
            attention_weights: None
        """
        gene_proj = self.U(pathway_mean)  # [rank]
        wsi_proj = self.V(wsi_embedding)    # [rank]
        bilinear = gene_proj * wsi_proj     # Element-wise [rank]

        combined = torch.cat([bilinear, pathway_mean, wsi_embedding], dim=-1)
        fused = self.projection(combined)

        return fused, None


class GatedFusion(nn.Module):
    """
    Gated fusion with learned modality weighting.

    Learns to weight contributions from each modality based on their content.
    Projects to output_dim first, then applies learned gates.
    """

    def __init__(self, gene_dim, wsi_dim, output_dim, dropout=0.3):
        super().__init__()

        # Project each modality to output_dim
        self.gene_proj = nn.Linear(gene_dim, output_dim)
        self.wsi_proj = nn.Linear(wsi_dim, output_dim)

        # Gate networks (operate on original dims for richer gating signal)
        self.gene_gate = nn.Sequential(
            nn.Linear(gene_dim, gene_dim // 2),
            nn.ReLU(),
            nn.Linear(gene_dim // 2, 1),
            nn.Sigmoid()
        )

        self.wsi_gate = nn.Sequential(
            nn.Linear(wsi_dim, wsi_dim // 2),
            nn.ReLU(),
            nn.Linear(wsi_dim // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        # Uniform interface — populated with [2] gate values each forward
        self.proto_pathway_gate_weights = None

    def forward(self, pathway_mean, wsi_embedding, pathway_embeddings=None, proto_tokens=None):
        """
        Args:
            pathway_mean: [hidden_dim] from gene encoder
            wsi_embedding: [hidden_dim] from WSI encoder

        Returns:
            fused: [output_dim] gated fused embedding
            gate_weights: [2] gene and WSI gate values
        """
        # Compute gates
        gene_gate = self.gene_gate(pathway_mean)  # [1]
        wsi_gate = self.wsi_gate(wsi_embedding)      # [1]

        # Normalize gates
        gate_sum = gene_gate + wsi_gate + 1e-6
        gene_weight = gene_gate / gate_sum
        wsi_weight = wsi_gate / gate_sum

        # Project and weight
        gene_proj = self.gene_proj(pathway_mean)
        wsi_proj = self.wsi_proj(wsi_embedding)

        fused = gene_weight * gene_proj + wsi_weight * wsi_proj
        fused = self.dropout(fused)

        gate_weights = torch.stack([gene_weight.squeeze(), wsi_weight.squeeze()])

        # Store for attention interface (detached, like CrossAttentionFusion)
        self.proto_pathway_gate_weights = gate_weights.detach()

        return fused, gate_weights


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_fusion_module(fusion_type, hidden_dim, output_dim=64, num_heads=4, dropout=0.3):
    """
    Factory function to create fusion module.

    Args:
        fusion_type: 'concat', 'cross_attention', 'simple_cross_attention', 'bilinear', 'gated'
        hidden_dim: Hidden dimension (assumes same for gene and WSI)
        output_dim: Output dimension for the fused embedding (fed to classifier)
        num_heads: Number of attention heads (for cross_attention)
        dropout: Dropout rate

    Returns:
        Fusion module instance
    """
    if fusion_type == 'concat':
        return ConcatFusion(hidden_dim, hidden_dim, output_dim, dropout)
    elif fusion_type == 'cross_attention':
        return CrossAttentionFusion(hidden_dim, output_dim, num_heads, dropout)
    elif fusion_type == 'simple_cross_attention':
        return SimpleCrossAttention(hidden_dim, output_dim, dropout)
    elif fusion_type == 'bilinear':
        return BilinearFusion(hidden_dim, hidden_dim, output_dim, rank=64, dropout=dropout)
    elif fusion_type == 'gated':
        return GatedFusion(hidden_dim, hidden_dim, output_dim, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")