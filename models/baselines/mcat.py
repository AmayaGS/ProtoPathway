"""
MCAT (Multimodal Co-Attention Transformer) Baseline for ProtoPathway.

Adapted from: https://github.com/mahmoodlab/MCAT

Key architecture:
1. Per-pathway SNNs (like SurvPath)
2. WSI projection
3. Co-attention: pathway queries attend to WSI patches (key/value)
4. Separate transformers for pathway and WSI branches
5. Gated attention pooling for each modality
6. Late fusion (concat or bilinear) + classifier

Changes from original:
- Uses PyG Data objects instead of separate x_omic1..6, x_path kwargs
- Lazy pathway network initialization from bipartite graph
- Returns only logits (survival hazard/S computed externally)
- Supports variable pathway counts from Reactome graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def SNN_Block(dim1: int, dim2: int, dropout: float = 0.25) -> nn.Sequential:
    """Self-Normalizing Network block."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


class Attn_Net_Gated(nn.Module):
    """
    Gated Attention Network (3 fc layers).

    Computes attention weights using sigmoid gating:
        a = tanh(W_a * x)
        b = sigmoid(W_b * x)
        A = W_c * (a ⊙ b)
    """

    def __init__(self, L: int = 256, D: int = 256, dropout: float = 0.25, n_classes: int = 1):
        super().__init__()

        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, L] input features

        Returns:
            A: [N, n_classes] attention logits (pre-softmax)
            x: [N, L] input features (pass-through)
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # element-wise gating
        return A, x


class BilinearFusion(nn.Module):
    """
    Bilinear fusion with gated multimodal units.

    From: https://github.com/mahmoodlab/MCAT/blob/master/models/model_utils.py
    """

    def __init__(
        self,
        dim1: int = 256,
        dim2: int = 256,
        scale_dim1: int = 8,
        scale_dim2: int = 8,
        mmhid: int = 256,
        dropout_rate: float = 0.25,
        skip: bool = False,
        use_bilinear: bool = False,
        gate1: bool = True,
        gate2: bool = True
    ):
        super().__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og = dim1, dim2
        dim1 = dim1 // scale_dim1
        dim2 = dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0

        # Gating for modality 1
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        if use_bilinear:
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1)
        else:
            self.linear_z1 = nn.Linear(dim1_og + dim2_og, dim1)
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Gating for modality 2
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        if use_bilinear:
            self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2)
        else:
            self.linear_z2 = nn.Linear(dim1_og + dim2_og, dim2)
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # Fusion layers
        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256 + skip_dim, mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vec1: [B, dim1] modality 1 features
            vec2: [B, dim2] modality 2 features

        Returns:
            fused: [B, mmhid] fused features
        """
        # Gated multimodal unit for modality 1
        if self.gate1:
            h1 = self.linear_h1(vec1)
            if self.use_bilinear:
                z1 = self.linear_z1(vec1, vec2)
            else:
                z1 = self.linear_z1(torch.cat([vec1, vec2], dim=-1))
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(self.linear_h1(vec1))

        # Gated multimodal unit for modality 2
        if self.gate2:
            h2 = self.linear_h2(vec2)
            if self.use_bilinear:
                z2 = self.linear_z2(vec1, vec2)
            else:
                z2 = self.linear_z2(torch.cat([vec1, vec2], dim=-1))
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(self.linear_h2(vec2))

        # Bilinear fusion: outer product with bias terms
        # Append 1 for bias: [B, dim+1]
        o1 = torch.cat([o1, torch.ones(o1.shape[0], 1, device=o1.device)], dim=-1)
        o2 = torch.cat([o2, torch.ones(o2.shape[0], 1, device=o2.device)], dim=-1)

        # Outer product: [B, dim1+1, dim2+1] -> [B, (dim1+1)*(dim2+1)]
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat([out, vec1, vec2], dim=-1)

        out = self.encoder2(out)
        return out


class MultiheadCoAttention(nn.Module):
    """
    Cross-attention where queries come from one modality (omic)
    and keys/values come from another (path).

    This is the core MCAT mechanism: pathway tokens query WSI patches.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Separate projections for Q (from omic) and K,V (from path)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: query attends to key/value.

        Args:
            query: [L, B, E] - omic tokens (L = num_pathways)
            key: [S, B, E] - path tokens (S = num_patches)
            value: [S, B, E] - path tokens

        Returns:
            output: [L, B, E] - attended omic tokens
            attn_weights: [B, num_heads, L, S] - attention weights
        """
        L, B, E = query.shape
        S = key.shape[0]

        # Project
        q = self.q_proj(query)  # [L, B, E]
        k = self.k_proj(key)    # [S, B, E]
        v = self.v_proj(value)  # [S, B, E]

        # Reshape for multi-head: [L, B, E] -> [B*H, L, head_dim]
        q = q.transpose(0, 1).contiguous().view(B * self.num_heads, L, self.head_dim)
        k = k.transpose(0, 1).contiguous().view(B * self.num_heads, S, self.head_dim)
        v = v.transpose(0, 1).contiguous().view(B * self.num_heads, S, self.head_dim)

        # Attention: [B*H, L, S]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights_raw = attn_weights.view(B, self.num_heads, L, S)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: [B*H, L, head_dim]
        output = torch.bmm(attn_weights, v)

        # Reshape back: [B*H, L, head_dim] -> [L, B, E]
        output = output.view(B, self.num_heads, L, self.head_dim)
        output = output.transpose(1, 2).contiguous().view(B, L, E)
        output = output.transpose(0, 1)  # [L, B, E]

        output = self.out_proj(output)

        return output, attn_weights_raw


class MCAT(nn.Module):
    """
    Multimodal Co-Attention Transformer.

    Architecture:
        1. Per-pathway SNN: pathway_genes -> hidden_dim
        2. WSI projection: patch_features -> hidden_dim
        3. Co-attention: pathway queries attend to WSI patches
        4. Path transformer: process WSI features (informed by co-attention)
        5. Omic transformer: process pathway features
        6. Gated attention pooling for each modality
        7. Fusion (concat or bilinear) + classifier

    Args:
        num_features: WSI patch feature dimension
        hidden_dim: Internal embedding dimension (256 in original)
        num_heads: Attention heads for transformers
        n_classes: Output classes (survival bins)
        dropout: Dropout rate
        fusion: 'concat' or 'bilinear'
        n_transformer_layers: Number of transformer encoder layers
    """

    def __init__(
        self,
        num_features: int = 1536,
        hidden_dim: int = 256,
        num_heads: int = 8,
        n_classes: int = 4,
        dropout: float = 0.25,
        fusion: str = 'concat',
        n_transformer_layers: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.fusion_type = fusion

        # --- WSI Projection ---
        self.wsi_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # --- Co-Attention (pathway queries, WSI keys/values) ---
        self.coattn = MultiheadCoAttention(
            embed_dim=hidden_dim,
            num_heads=1,  # Original uses 1 head for co-attention
            dropout=dropout
        )

        # --- Path Transformer ---
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Expects [batch, seq, dim]
        )
        self.path_transformer = nn.TransformerEncoder(
            path_encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False
        )

        # --- Omic Transformer ---
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Expects [batch, seq, dim]
        )
        self.omic_transformer = nn.TransformerEncoder(
            omic_encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False
        )

        # --- Gated Attention Pooling ---
        self.path_attention_head = Attn_Net_Gated(
            L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1
        )
        self.path_rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.omic_attention_head = Attn_Net_Gated(
            L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1
        )
        self.omic_rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Fusion ---
        if fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        elif fusion == 'bilinear':
            self.mm = BilinearFusion(
                dim1=hidden_dim,
                dim2=hidden_dim,
                scale_dim1=8,
                scale_dim2=8,
                mmhid=hidden_dim
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # --- Lazy-init pathway networks ---
        self.sig_networks = None
        self.num_pathways = None
        self._pathway_gene_indices = None

        # Store attention weights for interpretability
        self._attn_coattn = None
        self._attn_path = None
        self._attn_omic = None

    def _init_pathway_networks(self, data):
        """Build per-pathway SNNs on first forward pass."""
        pathway_gene_indices = data.pathway_gene_indices

        # Handle batched case (list of lists)
        if isinstance(pathway_gene_indices[0], list) and isinstance(pathway_gene_indices[0][0], list):
            pathway_gene_indices = pathway_gene_indices[0]

        self.num_pathways = len(pathway_gene_indices)

        # Store as tensors for efficient indexing
        self._pathway_gene_indices = [
            torch.tensor(indices, dtype=torch.long, device=data.x.device)
            for indices in pathway_gene_indices
        ]

        # Build per-pathway SNNs (matches original MCAT)
        sig_networks = []
        for indices in self._pathway_gene_indices:
            input_dim = len(indices)
            fc = nn.Sequential(
                SNN_Block(input_dim, self.hidden_dim, dropout=0.0),  # First block no dropout
                SNN_Block(self.hidden_dim, self.hidden_dim, dropout=0.25),
            )
            sig_networks.append(fc)

        self.sig_networks = nn.ModuleList(sig_networks).to(data.x.device)

    def forward(self, data, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: PyG Data object with:
                - x: [num_nodes, 1] gene expression (first num_genes rows)
                - wsi_features: [num_patches, wsi_dim]
                - pathway_gene_indices: list of gene index lists per pathway
                - num_genes, num_pathways

        Returns:
            logits: [1, n_classes]
        """
        # Lazy init pathway networks
        if self.sig_networks is None:
            self._init_pathway_networks(data)

        # === Extract gene expression ===
        num_genes = int(data.num_genes.item()) if torch.is_tensor(data.num_genes) else int(data.num_genes)
        gene_x = data.x[:num_genes].squeeze(-1)  # [num_genes]

        # === Pathway Branch: Per-pathway SNNs ===
        h_omic = []
        for idx, sig_net in enumerate(self.sig_networks):
            gene_indices = self._pathway_gene_indices[idx]
            pathway_genes = gene_x[gene_indices]  # [genes_in_pathway]
            pathway_embed = sig_net(pathway_genes.unsqueeze(0))  # [1, hidden_dim]
            h_omic.append(pathway_embed)

        # Stack: [1, num_pathways, hidden_dim] for batch_first format [batch, seq, dim]
        h_omic_bag = torch.stack(h_omic, dim=1)  # [1, P, hidden_dim]

        # === WSI Branch ===
        wsi_features = data.wsi_features
        if wsi_features.dim() == 3:
            wsi_features = wsi_features.squeeze(0)

        # Project WSI patches
        h_path_bag = self.wsi_net(wsi_features)  # [num_patches, hidden_dim]
        h_path_bag = h_path_bag.unsqueeze(0)  # [1, S, hidden_dim] for batch_first format

        # === Co-Attention: Pathways query WSI patches ===
        # Query: pathway embeddings, Key/Value: WSI embeddings
        # Co-attention expects [seq, batch, dim] format
        h_path_coattn, A_coattn = self.coattn(
            h_omic_bag.transpose(0, 1),  # [P, 1, hidden_dim]
            h_path_bag.transpose(0, 1),  # [S, 1, hidden_dim]
            h_path_bag.transpose(0, 1)   # [S, 1, hidden_dim]
        )
        # h_path_coattn: [P, 1, hidden_dim] -> convert to batch_first
        h_path_coattn = h_path_coattn.transpose(0, 1)  # [1, P, hidden_dim]

        if return_attention:
            self._attn_coattn = A_coattn.detach()

        # === Path Transformer ===
        # Process the co-attended features through path transformer
        h_path_trans = self.path_transformer(h_path_coattn)  # [1, P, hidden_dim]

        # Gated attention pooling over pathway dimension
        h_path_trans_squeezed = h_path_trans.squeeze(0)  # [P, hidden_dim]
        A_path, h_path_feat = self.path_attention_head(h_path_trans_squeezed)
        A_path = A_path.transpose(1, 0)  # [1, P]
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path_feat)  # [1, hidden_dim]
        h_path = self.path_rho(h_path).squeeze(0)  # [hidden_dim]

        if return_attention:
            self._attn_path = A_path.detach()

        # === Omic Transformer ===
        h_omic_trans = self.omic_transformer(h_omic_bag)  # [1, P, hidden_dim]

        # Gated attention pooling
        h_omic_trans_squeezed = h_omic_trans.squeeze(0)  # [P, hidden_dim]
        A_omic, h_omic_feat = self.omic_attention_head(h_omic_trans_squeezed)
        A_omic = A_omic.transpose(1, 0)  # [1, P]
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic_feat)  # [1, hidden_dim]
        h_omic = self.omic_rho(h_omic).squeeze(0)  # [hidden_dim]

        if return_attention:
            self._attn_omic = A_omic.detach()

        # === Fusion ===
        if self.fusion_type == 'bilinear':
            h = self.mm(h_path.unsqueeze(0), h_omic.unsqueeze(0)).squeeze(0)
        else:  # concat
            h = self.mm(torch.cat([h_path, h_omic], dim=0))

        # === Classifier ===
        logits = self.classifier(h).unsqueeze(0)  # [1, n_classes]

        return logits

    def get_attention_outputs(self):
        """Return stored attention weights for interpretability."""
        return {
            'coattn': self._attn_coattn,  # [1, 1, P, S] pathway-to-patch attention
            'path': self._attn_path,       # [1, P] path pooling attention
            'omic': self._attn_omic,       # [1, P] omic pooling attention
            'num_pathways': self.num_pathways,
        }