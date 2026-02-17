"""
MMP (MultiModal Prototyping) Baseline - Corrected Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Network block."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.SELU(inplace=True),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


class PANTHERAggregator(nn.Module):
    """
    True PANTHER-style aggregation with EM iterations.
    Output: [K, 1 + 2*D] = [pi, mu, sigma] per prototype
    """

    def __init__(self, centroids: torch.Tensor, n_em_iters: int = 3, tau: float = 10.0, eps: float = 0.1):
        super().__init__()
        self.register_buffer("centroids", centroids)  # [K, D] - prior means
        self.K, self.D = centroids.shape
        self.n_em_iters = n_em_iters
        self.tau = tau
        self.eps = eps

        # Prior covariance (learnable or fixed)
        self.register_buffer("prior_cov", eps * torch.ones(self.K, self.D))

    def forward(self, patches: torch.Tensor):
        """
        patches: [N, D]
        returns: [K, 1 + 2*D] = [pi | mu | sigma]
        """
        N, D = patches.shape

        # Initialize from prior
        pi = torch.ones(self.K, device=patches.device) / self.K
        mu = self.centroids.clone()
        sigma = self.prior_cov.clone()

        # EM iterations
        for _ in range(self.n_em_iters):
            # E-step: compute responsibilities
            # log p(x|k) = -0.5 * [D*log(2π) + sum(log σ) + sum((x-μ)²/σ)]
            diff = patches.unsqueeze(1) - mu.unsqueeze(0)  # [N, K, D]
            log_prob = -0.5 * (
                    D * torch.log(torch.tensor(2 * 3.14159, device=patches.device)) +
                    sigma.log().sum(-1) +  # [K]
                    (diff ** 2 / sigma).sum(-1)  # [N, K]
            )
            log_resp = log_prob + pi.log()  # [N, K]
            log_resp = log_resp - log_resp.logsumexp(dim=1, keepdim=True)
            resp = log_resp.exp()  # [N, K]

            # M-step with prior regularization
            Nk = resp.sum(0) + self.tau  # [K]
            pi = Nk / Nk.sum()

            # Weighted mean with prior
            weighted_sum = resp.T @ patches  # [K, D]
            mu = (weighted_sum + self.tau * self.centroids) / Nk.unsqueeze(1)

            # Weighted covariance with prior
            weighted_sq_sum = resp.T @ (patches ** 2)  # [K, D]
            sigma = (weighted_sq_sum + self.tau * (self.prior_cov + self.centroids ** 2)) / Nk.unsqueeze(1) - mu ** 2
            sigma = sigma.clamp(min=1e-6)  # numerical stability

        # Concatenate [pi | mu | sigma] -> [K, 1 + 2*D]
        summary = torch.cat([pi.unsqueeze(1), mu, sigma], dim=1)
        return summary


class FeedForward(nn.Module):
    """Feed-forward network matching original MMP."""

    def __init__(self, dim, mult=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MMAttentionLayer(nn.Module):
    """
    Multimodal attention layer matching original MMP.
    Output dimension is halved (dim -> dim // 2).
    """

    def __init__(self, dim, dim_head, heads=1, dropout=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim_head * 3, bias=False)
        self.to_out = nn.Linear(dim_head, dim_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: [B, N, dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv  # each [B, N, dim_head]

        # Attention
        scale = self.dim_head ** -0.5
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, N, dim_head]
        out = self.to_out(out)

        if return_attention:
            return out, attn
        return out


class MMP(nn.Module):
    """
    MultiModal Prototyping - Corrected to match original implementation.
    """

    def __init__(
            self,
            centroids: torch.Tensor = None,
            num_features: int = 1536,
            hidden_dim: int = 256,
            num_heads: int = 1,
            n_classes: int = 4,
            dropout: float = 0.1,
            n_em_iters: int = 3,
            tau: float = 10.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.out_dim = hidden_dim // 2

        # Lazy init for pathway networks
        self.sig_networks = None
        self.num_pathways = None
        self._pathway_gene_indices = None

        # PANTHER aggregator
        if centroids is not None:
            self.panther_aggregator = PANTHERAggregator(
                centroids, n_em_iters=n_em_iters, tau=tau
            )
            self.n_proto = centroids.shape[0]
            self.wsi_feat_dim = centroids.shape[1]
            # Input dim = 1 + 2*D (pi, mu, sigma per prototype)
            self.histo_proj = nn.Linear(1 + 2 * self.wsi_feat_dim, hidden_dim)
        else:
            self.panther_aggregator = None
            self.n_proto = None
            self.wsi_feat_dim = num_features
            self.histo_proj = None

        # Attention layer
        self.cross_attention = MMAttentionLayer(
            dim=hidden_dim,
            dim_head=self.out_dim,
            heads=num_heads,
            dropout=dropout
        )

        # Feed-forward and layer norm (on halved dimension)
        self.feed_forward = FeedForward(self.out_dim, mult=1, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.out_dim)

        # Final classifier: concat of [pathway_pool, histo_pool]
        self.classifier = nn.Linear(self.out_dim * 2, n_classes)

        self._attn_weights = None

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

        # Per-pathway SNNs (matches original exactly)
        sig_networks = []
        for indices in self._pathway_gene_indices:
            input_dim = len(indices)
            fc = nn.Sequential(
                SNN_Block(input_dim, self.hidden_dim, dropout=0.0),  # First block no dropout
                SNN_Block(self.hidden_dim, self.hidden_dim, dropout=0.25),
            )
            sig_networks.append(fc)
        self.sig_networks = nn.ModuleList(sig_networks).to(data.x.device)

    def forward(self, data, return_attention: bool = False):
        if self.sig_networks is None:
            self._init_pathway_networks(data)

        if self.panther_aggregator is None:
            raise ValueError("MMP requires pre-computed centroids.")

        # === Histology Branch: PANTHER Aggregation ===
        wsi_features = data.wsi_features
        if wsi_features.dim() == 3:
            wsi_features = wsi_features.squeeze(0)

        # PANTHER -> [K, 1 + 2*D]
        slide_summary = self.panther_aggregator(wsi_features)
        h_histo = self.histo_proj(slide_summary).unsqueeze(0)  # [1, K, hidden_dim]

        # === Pathway Branch: Per-pathway SNNs ===
        num_genes = int(data.num_genes.item()) if torch.is_tensor(data.num_genes) else int(data.num_genes)
        gene_x = data.x[:num_genes].squeeze(-1)

        h_omic = []
        for idx, sig_net in enumerate(self.sig_networks):
            gene_indices = self._pathway_gene_indices[idx]
            pathway_genes = gene_x[gene_indices]
            pathway_embed = sig_net(pathway_genes.unsqueeze(0))
            h_omic.append(pathway_embed)

        h_omic = torch.stack(h_omic, dim=1)  # [1, P, hidden_dim]

        # === Concatenate: [omic | histo]  ===
        tokens = torch.cat([h_omic, h_histo], dim=1)  # [1, P + K, hidden_dim]

        # === Attention ===
        if return_attention:
            attn_out, self._attn_weights = self.cross_attention(tokens, return_attention=True)
        else:
            attn_out = self.cross_attention(tokens)  # [1, P + K, out_dim]

        # === Feed-forward + LayerNorm ===
        ff_out = self.feed_forward(attn_out)
        tokens_out = self.layer_norm(ff_out)  # [1, P + K, out_dim]

        # === Modality-specific pooling  ===
        pathway_embed = tokens_out[:, :self.num_pathways, :].mean(dim=1)  # [1, out_dim]
        histo_embed = tokens_out[:, self.num_pathways:, :].mean(dim=1)  # [1, out_dim]

        # === Concatenate and classify ===
        embedding = torch.cat([pathway_embed, histo_embed], dim=1)  # [1, out_dim * 2]
        logits = self.classifier(embedding)

        return logits

    def get_attention_outputs(self):
        return {
            'multimodal_attention': self._attn_weights,
            'n_histo_proto': self.n_proto,
            'n_pathway_proto': self.num_pathways,
        }