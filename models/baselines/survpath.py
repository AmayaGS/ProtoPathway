import torch
import torch.nn as nn


def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Network block (from original SurvPath)."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


class FeedForward(nn.Module):
    """Feed-forward block for transformer."""

    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MMAttentionLayer(nn.Module):
    """
    Multimodal attention layer from SurvPath.
    Self-attention over concatenated pathway + WSI tokens.
    """

    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)
        self._attn_weights = None

    def forward(self, x, return_attention=False):
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        if return_attention:
            self._attn_weights = attn_weights.detach()

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x


class SurvPath(nn.Module):
    """
    SurvPath: Separate MLP per pathway + multimodal transformer.

    1. Each pathway has dedicated SNN: genes_in_pathway -> 256 -> 256
    2. WSI patches projected to same dimension
    3. Multimodal self-attention over all tokens
    4. Modality-specific mean pooling -> concat -> classifier

    Args:
        omic_sizes: List of gene counts per pathway (from bipartite graph)
        num_features: Dimension of WSI patch features
        hidden_dim: Pathway embedding dimension (256 in original)
        num_heads: Attention heads
        n_classes: Output classes (survival bins)
        dropout: Dropout rate
    """

    def __init__(
            self,
            num_features: int = 1536,
            hidden_dim: int = 256,
            num_heads: int = 1,
            n_classes: int = 4,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.sig_networks = None

        # --- WSI projection ---
        self.wsi_projection = nn.Linear(num_features, hidden_dim)

        # --- Multimodal transformer ---
        self.cross_attender = MMAttentionLayer(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # --- Post-attention processing ---
        self.feed_forward = FeedForward(hidden_dim, mult=2, dropout=dropout)
        # Project down before layer norm
        self.post_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        self.layer_norm = nn.LayerNorm(hidden_dim // 2)

        # --- Classifier (pathway_pooled + wsi_pooled) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_classes),
        )

        # Storage for attention
        self._attn_pathways = None

        # Will be set during first forward pass
        self._pathway_gene_indices = None

    def _init_pathway_networks(self, data):
        """Build per-pathway MLPs on first forward pass."""
        pathway_gene_indices = data.pathway_gene_indices
        if isinstance(pathway_gene_indices[0][0], list):
            pathway_gene_indices = pathway_gene_indices[0]
        self.num_pathways = len(pathway_gene_indices)

        self._pathway_gene_indices = [
            torch.tensor(indices, dtype=torch.long, device=data.x.device)
            for indices in pathway_gene_indices
        ]

        # Build MLPs with correct input dims
        sig_networks = []
        for indices in self._pathway_gene_indices:
            input_dim = len(indices)
            fc = nn.Sequential(
                SNN_Block(input_dim, self.hidden_dim),
                SNN_Block(self.hidden_dim, self.hidden_dim),
            )
            sig_networks.append(fc)
        self.sig_networks = nn.ModuleList(sig_networks).to(data.x.device)


    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data with:
                - x: [num_nodes, 1] gene expression (first num_genes rows)
                - edge_index: bipartite graph
                - num_genes, num_pathways
                - wsi_features: [num_patches, wsi_dim]

        Returns:
            logits: [1, n_classes]
        """
        # Build pathway->gene mapping on first call
        if self.sig_networks is None:
            self._init_pathway_networks(data)

        # Rest stays exactly the same
        gene_x = data.x[:data.num_genes].squeeze(-1)

        h_omic = []
        for idx, sig_net in enumerate(self.sig_networks):
            gene_indices = self._pathway_gene_indices[idx]
            pathway_genes = gene_x[gene_indices]
            pathway_embed = sig_net(pathway_genes.unsqueeze(0))
            h_omic.append(pathway_embed)

        h_omic_bag = torch.cat(h_omic, dim=0).unsqueeze(0)  # [1, num_pathways, hidden_dim]

        # --- WSI embeddings ---
        wsi_features = data.wsi_features
        if wsi_features.dim() == 3:
            wsi_features = wsi_features.squeeze(0)
        wsi_embed = self.wsi_projection(wsi_features).unsqueeze(0)  # [1, num_patches, hidden_dim]

        # --- Concatenate tokens ---
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)  # [1, P+N, hidden_dim]

        # --- Multimodal attention ---
        mm_embed = self.cross_attender(tokens, return_attention=return_attention)

        if return_attention:
            self._attn_pathways = self.cross_attender._attn_weights

        # --- Feed forward + layer norm ---
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.post_proj(mm_embed)  # [1, P+N, hidden_dim//2]
        mm_embed = self.layer_norm(mm_embed)

        # --- Modality-specific pooling ---
        pathway_embed = mm_embed[:, :self.num_pathways, :]  # [1, P, D//2]
        wsi_embed = mm_embed[:, self.num_pathways:, :]  # [1, N, D//2]

        pathway_pooled = pathway_embed.mean(dim=1)  # [1, D//2]
        wsi_pooled = wsi_embed.mean(dim=1)  # [1, D//2]

        # --- Fuse and classify ---
        fused = torch.cat([pathway_pooled, wsi_pooled], dim=1)  # [1, D]
        logits = self.classifier(fused)  # [1, n_classes]

        return logits

    def get_attention_outputs(self):
        return {
            'multimodal_attention': self._attn_pathways,
            'num_pathways': self.num_pathways,
        }