"""
PIBD Baseline for ProtoPathway.

From: Zhang et al. "Prototypical Information Bottlenecking and Disentangling
for Multimodal Cancer Survival Prediction" (ICLR 2024)

This file includes all necessary components:
- SNN_Block: Self-normalizing network block
- MIEstimator (CLUB): Mutual information estimator
- MITransformerLayer: Disentanglement transformer
- PIB: Prototypical Information Bottleneck
- PoE: Product of Experts
- PIBD: Main model

Adapted to work with ProtoPathway's PyG data interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import NLLSurvLoss

from models.baselines.pibd.distentangle_transformer import MITransformerLayer
from models.baselines.pibd.club import MIEstimator


# =============================================================================
# Basic Building Blocks
# =============================================================================

def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Network block."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


# =============================================================================
# Product of Experts (PoE)
# =============================================================================

class PoE(nn.Module):
    """Product of Experts for fusing multimodal distributions."""

    def __init__(self, modality_num=2, sample_num=50, seed=1):
        super().__init__()
        self.sample_num = sample_num
        self.seed = seed

        phi = torch.ones(modality_num, requires_grad=True)
        self.phi = nn.Parameter(phi)

    def forward(self, mu_list, var_list, eps=1e-8):
        t_sum = 0
        mu_t_sum = 0

        alpha = F.softmax(self.phi, dim=0)

        for idx, (mu, var) in enumerate(zip(mu_list, var_list)):
            T = 1 / (var + eps)
            t_sum += alpha[idx] * T
            mu_t_sum += mu * alpha[idx] * T

        mu = mu_t_sum / t_sum
        var = 1 / t_sum

        dim = mu.shape[1]
        batch_size = mu.shape[0]
        eps_noise = self._gaussian_noise(samples=(batch_size, self.sample_num), k=dim)
        poe_features = mu.unsqueeze(1) + var.unsqueeze(1) * eps_noise

        return poe_features

    def _gaussian_noise(self, samples, k):
        if self.training:
            return torch.normal(
                torch.zeros(*samples, k),
                torch.ones(*samples, k)
            ).cuda()
        else:
            return torch.normal(
                torch.zeros(*samples, k),
                torch.ones(*samples, k),
                generator=torch.manual_seed(self.seed)
            ).cuda()


# =============================================================================
# Prototypical Information Bottleneck (PIB)
# =============================================================================

def KL_between_normals(q_distr, p_distr):
    """KL divergence between two normal distributions."""
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + \
         torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


class PIB(nn.Module):
    """Prototypical Information Bottleneck."""

    def __init__(self, x_dim, z_dim=256, num_classes=4, topk=256, sample_num=50, seed=1):
        super().__init__()

        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )

        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # Proxies: 2 per class (censored/uncensored)
        self.proxies = nn.Parameter(torch.empty([num_classes * 2, z_dim * 2]))
        nn.init.xavier_uniform_(self.proxies, gain=1.0)

        # Mapping: "censor,label" -> index
        self.proxies_dict = {
            f"{c},{l}": c * num_classes + l
            for c in range(2) for l in range(num_classes)
        }

    def _gaussian_noise(self, samples, k):
        if self.training:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k)).cuda()
        else:
            return torch.normal(
                torch.zeros(*samples, k), torch.ones(*samples, k),
                generator=torch.manual_seed(self.seed)
            ).cuda()

    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = F.softplus(self.proxies[:, self.z_dim:])
        return mu_proxy, sigma_proxy

    def forward(self, x, y=None, c=None):
        """
        Args:
            x: [B, N, x_dim] - input features
            y: [B] - survival bin labels (None for inference)
            c: [B] - censorship (None for inference)
        """
        feature_num = x.shape[1]

        # Encode
        z = self.encoder(x)  # [B, N, z_dim]

        # Get proxy distributions
        mu_proxy, sigma_proxy = self.encoder_proxies()

        # Sample from proxies
        eps_proxy = self._gaussian_noise([self.num_classes * 2, self.sample_num], self.z_dim)
        z_proxy_sample = mu_proxy.unsqueeze(1) + sigma_proxy.unsqueeze(1) * eps_proxy
        z_proxy = torch.mean(z_proxy_sample, dim=1)

        # Attention between features and proxies
        z_norm = F.normalize(z, dim=2)
        z_proxy_norm = F.normalize(z_proxy).unsqueeze(0)
        att = torch.matmul(z_norm, z_proxy_norm.transpose(1, 2))

        if y is None and c is None:
            # Inference: find best proxy
            att_unbind = torch.cat(torch.unbind(att, dim=1), dim=1)
            _, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
            att_topk_idx = att_topk_idx % (self.num_classes * 2)
            positive_proxy_idx, _ = torch.mode(att_topk_idx, dim=1)
            positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(1)
            proxy_loss = None
        else:
            # Training: use ground truth proxy
            proxy_indices = torch.tensor([
                self.proxies_dict[f"{int(c_i)},{int(y_i)}"]
                for c_i, y_i in zip(c, y)
            ], device=x.device).long()

            mask = torch.zeros_like(att, dtype=torch.bool)
            mask[torch.arange(att.size(0)), :, proxy_indices] = True

            att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
            att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)

            # Clamp topk to available features
            k_pos = min(self.topk, att_positive.size(1))
            k_neg = min(self.topk, att_negative.size(1))

            att_topk_positive, _ = torch.topk(att_positive.squeeze(2), k_pos, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, k_neg, dim=1)

            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1), dim=1)
            proxy_loss = -(att_positive_mean - att_negative_mean).mean()

            positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(1)

        # Gather proxy params for each sample
        mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)
        sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
        mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(1)
        sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(1)

        # Get top-k features
        att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)
        k = min(self.topk, att_unbind.size(1))
        att_topk, att_topk_idx = torch.topk(att_unbind, k, dim=1)
        att_topk_idx = att_topk_idx % feature_num
        z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(2).repeat(1, 1, self.z_dim))

        decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)

        return decoder_logits_proxy, mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk, proxy_loss


# =============================================================================
# Main PIBD Model
# =============================================================================

class PIBD(nn.Module):
    """
    Prototypical Information Bottlenecking and Disentangling.

    Adapted to ProtoPathway's data interface with lazy initialization
    for per-pathway MLPs.
    """

    def __init__(self, num_features: int = 1536,
                 hidden_dim: int = 256,
                 n_classes: int = 4,
                 bag_size: int = 512,
                 num_patches: int = 4096,
                 ratio_wsi: float = 0.5,
                 ratio_omics: float = 0.5,
                 sample_num: int = 50,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 seed: int = 42):
        super().__init__()

        self.num_features = num_features
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.bag_size = bag_size
        self.ratio_wsi = ratio_wsi
        self.ratio_omics = ratio_omics
        self.sample_num = sample_num
        self.alpha = alpha
        self.beta = beta
        self.seed = seed

        # Lazy init - built on first forward
        self.sig_networks = None
        self.num_pathways = None

        # PIB modules - WSI built on first forward (needs num_features confirmed)
        self.PIB_wsi = None
        self.PIB_omics = None

        # Disentanglement
        self.PID = MITransformerLayer(
            dim=hidden_dim,
            num_heads=4,
            mlp_ratio=1.,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path=0.1,
        )

        # Classifier
        self.to_logits = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

        # PoE
        self.PoE = PoE(modality_num=2, sample_num=sample_num, seed=seed)

        # CLUB MI estimator
        self.CLUB = MIEstimator(hidden_dim)

        self.loss_surv = NLLSurvLoss(alpha=0.5)

        # Attention storage
        self._attns = None

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

        sig_networks = []
        for indices in self._pathway_gene_indices:
            input_dim = len(indices)
            fc = nn.Sequential(
                SNN_Block(input_dim, self.hidden_dim),
                SNN_Block(self.hidden_dim, self.hidden_dim, dropout=0.25),
            )
            sig_networks.append(fc)
        self.sig_networks = nn.ModuleList(sig_networks).to(data.x.device)

        # Initialize PIB modules
        topk_wsi = int(self.bag_size * self.ratio_wsi)
        topk_omics = int(self.num_pathways * self.ratio_omics)

        self.PIB_wsi = PIB(
            self.num_features, self.hidden_dim,
            num_classes=self.n_classes, topk=topk_wsi,
            sample_num=self.sample_num, seed=self.seed
        ).to(data.x.device)

        self.PIB_omics = PIB(
            self.hidden_dim, self.hidden_dim,
            num_classes=self.n_classes, topk=topk_omics,
            sample_num=self.sample_num, seed=self.seed
        ).to(data.x.device)

    def _get_loss_proxy(self, decoder_logits, y, c):
        """
        Proxy survival loss (PIB-style), adapted to NLLSurvLoss(event=1 means death).
        """

        device = decoder_logits.device
        n = self.n_classes

        # Fake batch of size (2 * n_classes)
        # First half: uncensored (event=1)
        # Second half: censored (event=0)
        event = torch.zeros(2 * n, device=device)
        event[:n] = 1.0  # death occurred

        # Time-bin labels: [0, 1, ..., n-1, 0, 1, ..., n-1]
        y_proxy = torch.arange(n, device=device).repeat(2)

        return self.loss_surv(
            decoder_logits,  # h
            y_proxy,  # y
            None,  # t (unused, but required)
            event  # event (NOT c)
        )


    def _get_KL_loss(self, mu, std):
        """KL divergence from standard normal."""
        prior = (torch.zeros_like(mu), torch.ones_like(std))
        posterior = (mu, std)
        return torch.mean(KL_between_normals(posterior, prior))

    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data with:
                - x: [num_nodes, 1] gene expression
                - wsi_features: [N, num_features]
                - pathway_gene_indices: list of gene index lists
                - y: {'bin': label, 'event': censorship} during training

        Returns:
            If training: (logits, aux_loss_dict)
            If eval: logits
        """
        # Lazy init
        if self.sig_networks is None:
            self._init_pathway_networks(data)

        # Get gene expression
        num_genes = int(data.num_genes.item()) if torch.is_tensor(data.num_genes) else int(data.num_genes)
        gene_x = data.x[:num_genes].squeeze(-1)

        # Encode pathways
        h_omic = []
        for idx, sig_net in enumerate(self.sig_networks):
            gene_indices = self._pathway_gene_indices[idx]
            pathway_genes = gene_x[gene_indices]
            pathway_embed = sig_net(pathway_genes.unsqueeze(0))
            h_omic.append(pathway_embed)
        h_omic_bag = torch.stack(h_omic, dim=1)  # [1, num_pathways, hidden_dim]

        # Get WSI features
        wsi_features = data.wsi_features
        if wsi_features.dim() == 2:
            wsi_features = wsi_features.unsqueeze(0)  # [1, N, D]

        # Get labels for training
        if self.training and hasattr(data, 'y') and isinstance(data.y, dict):
            y = data.y['bin'].unsqueeze(0) if data.y['bin'].dim() == 0 else data.y['bin']
            event = data.y['event'].unsqueeze(0) if data.y['event'].dim() == 0 else data.y['event']
            c = 1 - event
        else:
            y = None
            c = None

        # PIB for omics
        (decoder_logits_proxy_omic, mu_proxy_omic, sigma_proxy_omic,
         h_omic_bag, mu_topk_omic, sigma_topk_omic, proxy_loss_omic) = self.PIB_omics(h_omic_bag, y, c)

        # Subsample WSI patches
        actual_num_patches = wsi_features.size(1)
        num_iters = self.num_patches // self.bag_size

        logits_list = []
        mimin_total = 0.0
        mimin_loss_total = 0.0
        IB_loss_total = 0.0
        proxy_loss_total = 0.0

        for i in range(num_iters):
            # Random sample patches
            if self.num_patches >= self.bag_size:
                idx = torch.randperm(actual_num_patches, device=wsi_features.device)[:self.bag_size]
                h_wsi_bag = wsi_features[:, idx, :]
            else:
                h_wsi_bag = wsi_features

            # PIB for WSI
            (decoder_logits_proxy_wsi, mu_proxy_wsi, sigma_proxy_wsi,
             h_wsi_bag, mu_topk_wsi, sigma_topk_wsi, proxy_loss_wsi) = self.PIB_wsi(h_wsi_bag, y, c)

            # PoE fusion
            mu_list = [mu_topk_wsi, mu_topk_omic]
            var_list = [sigma_topk_wsi, sigma_topk_omic]
            poe_features = self.PoE(mu_list, var_list)
            poe_embed = torch.mean(poe_features, dim=1)
            poe_embed = poe_embed.unsqueeze(1).expand(h_wsi_bag.size(0), 1, -1)

            # Disentanglement
            if return_attention:
                histology, pathways, global_embed, attns = self.PID(
                    h_wsi_bag, h_omic_bag, poe_embed, return_attention=True
                )
                self._attns = attns
            else:
                histology, pathways, global_embed = self.PID(
                    h_wsi_bag, h_omic_bag, poe_embed
                )

            # Pool
            histology = torch.mean(histology, dim=1)
            pathways = torch.mean(pathways, dim=1)
            global_embed = torch.mean(global_embed, dim=1)

            # MI estimation
            mimin = self.CLUB(histology, pathways, global_embed)
            mimin_loss = self.CLUB.learning_loss(histology, pathways, global_embed)

            # Classify
            logits = self.to_logits(torch.cat([histology, pathways, global_embed], dim=-1))
            logits_list.append(logits)

            if self.training:
                mimin_total += mimin
                mimin_loss_total += mimin_loss
                IB_loss_total += (
                        self.alpha * self._get_loss_proxy(decoder_logits_proxy_wsi, y, c) +
                        self.alpha * self._get_loss_proxy(decoder_logits_proxy_omic, y, c) +
                        self.beta * self._get_KL_loss(mu_proxy_wsi, sigma_proxy_wsi) +
                        self.beta * self._get_KL_loss(mu_proxy_omic, sigma_proxy_omic)
                )
                if proxy_loss_wsi is not None:
                    proxy_loss_total += proxy_loss_wsi
                if proxy_loss_omic is not None:
                    proxy_loss_total += proxy_loss_omic

        # Average across iterations
        logits = torch.mean(torch.stack(logits_list), dim=0)

        if self.training:
            aux_losses = {
                'IB_loss': IB_loss_total / num_iters,
                'proxy_loss': proxy_loss_total / num_iters,
                'mimin': mimin_total / num_iters,
                'mimin_loss': mimin_loss_total / num_iters,
            }
            return logits, aux_losses

        return logits

    def get_attention_outputs(self):
        return {'disentangle_attention': self._attns}


