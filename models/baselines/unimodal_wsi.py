"""
WSI Baseline Models for ProtoPathway.

Implements:
- ABMIL: Attention-Based Multiple Instance Learning
- TransMIL: Transformer-based MIL

All models follow the same interface:
    logits = model(data)  # data.wsi_features: [num_patches, feature_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention



class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning.

    Reference:
        Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)

    Architecture:
        patches -> gated attention -> weighted sum -> classifier
    """

    def __init__(self, input_dim: int = 1536,
                 n_classes: int = 4,
                 attention_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Sigmoid()
        )

        self.attention_W = nn.Linear(attention_dim, 1)

        # Classifier
        self.classifier = nn.Linear(input_dim, n_classes)

        # Store attention for visualization
        self._attention_weights = None

    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data object with wsi_features [num_patches, input_dim]
            return_attention: Whether to store attention weights

        Returns:
            logits: [batch_size, n_classes]
        """
        # Handle batched or single input
        x = data.wsi_features
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, P, D]

        B, P, D = x.shape

        # Gated attention
        A_V = self.attention_V(x)  # [B, P, attention_dim]
        A_U = self.attention_U(x)  # [B, P, attention_dim]
        A = self.attention_W(A_V * A_U)  # [B, P, 1]
        A = A.squeeze(-1)  # [B, P]
        A = F.softmax(A, dim=1)  # [B, P]

        if return_attention:
            self._attention_weights = A.detach()

        # Weighted aggregation
        M = torch.bmm(A.unsqueeze(1), x)  # [B, 1, hidden_dim]
        M = M.squeeze(1)  # [B, hidden_dim]

        # Classification
        logits = self.classifier(M)  # [B, n_classes]

        return logits

    def get_attention_outputs(self):
        """Return stored attention weights."""
        return {'patch_attention': self._attention_weights}


class TransLayer(nn.Module):

    """
    code from - https://github.com/szc19990412/TransMIL

    """

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):

    def __init__(self, num_features=1536, n_classes=2):
        super(TransMIL, self).__init__()

        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, data, return_attention: bool = False):
        """
        Args:
            data.wsi_features:
                [P, num_features] or [B, P, num_features]

        Returns:
            logits: [B, n_classes]
        """
        h = data.wsi_features

        # ---- handle batching
        if h.dim() == 2:
            h = h.unsqueeze(0)  # [1, P, F]

        B, P, F = h.shape

        # ---- feature projection
        h = self._fc1(h)  # [B, P, 512]

        # ---- pad to square grid for PPEG
        H = h.shape[1]
        grid_size = int(np.ceil(np.sqrt(H)))
        target_len = grid_size * grid_size
        add_length = target_len - H

        if add_length > 0:
            h = torch.cat([h, h[:, :add_length]], dim=1)  # [B, target_len, 512]

        # ---- cls token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, 1 + target_len, 512]

        # ---- TransMIL blocks
        h = self.layer1(h)
        h = self.pos_layer(h, grid_size, grid_size)
        h = self.layer2(h)

        # ---- CLS pooling
        h = self.norm(h)[:, 0]  # [B, 512]

        # ---- prediction head
        logits = self._fc2(h)  # [B, n_classes]

        return logits

    def get_attention_outputs(self):
        """Return stored attention weights (if implemented)."""
        return {'transformer_attention': self._attention_weights}


class DSMIL(nn.Module):
    """
    Implementation of DSMIL (Li et al., CVPR 2021),
    adapted to the pipeline API:
      - input: data.wsi_features [P, D] or [B, P, D]
      - output: logits [B, n_classes]
    """

    def __init__(
        self,
        input_dim: int = 1536,
        n_classes: int = 4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        # ---- Instance classifier (linear, as in paper)
        self.instance_classifier = nn.Linear(input_dim, n_classes)

        # ---- Feature projection for bag stream
        # (called "fc" in the official repo)
        self.feature_proj = nn.Linear(input_dim, input_dim)

        self._attention_weights = None
        self._instance_scores = None

    def forward(self, data, return_attention: bool = False):
        """
        Args:
            data.wsi_features: [P, D] or [B, P, D]

        Returns:
            logits: [B, n_classes]
        """
        x = data.wsi_features
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, P, D]

        B, P, D = x.shape

        # ---- Instance-level stream
        x_flat = x.view(-1, D)  # [B*P, D]
        instance_logits = self.instance_classifier(x_flat)
        instance_logits = instance_logits.view(B, P, self.n_classes)  # [B, P, C]

        # ---- Find critical instances per class
        # index of max-scoring patch for each class
        critical_idx = instance_logits.argmax(dim=1)  # [B, C]

        # ---- Feature projection
        h = self.feature_proj(x_flat).view(B, P, D)  # [B, P, D]

        # ---- Class-wise attention & aggregation
        bag_logits = []

        for c in range(self.n_classes):
            # Attention weights from instance logits
            A_c = instance_logits[:, :, c]  # [B, P]
            A_c = F.softmax(A_c, dim=1)

            # Bag representation for class c
            M_c = torch.bmm(A_c.unsqueeze(1), h).squeeze(1)  # [B, D]

            # Classify using instance classifier weights (weight sharing!)
            w_c = self.instance_classifier.weight[c]  # [D]
            b_c = self.instance_classifier.bias[c]    # scalar

            logit_c = (M_c * w_c).sum(dim=1) + b_c  # [B]
            bag_logits.append(logit_c)

        logits = torch.stack(bag_logits, dim=1)  # [B, C]

        if return_attention:
            self._attention_weights = instance_logits.softmax(dim=1).detach()
            self._instance_scores = instance_logits.detach()

        return logits

    def get_attention_outputs(self):
        return {
            "patch_attention": self._attention_weights,
            "instance_scores": self._instance_scores
        }
