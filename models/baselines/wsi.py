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



class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning.

    Reference:
        Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)

    Architecture:
        patches -> FC -> tanh -> attention_scores -> weighted_sum -> classifier
    """

    def __init__(
            self,
            input_dim: int = 1536,
            hidden_dim: int = 256,
            n_classes: int = 4,
            dropout: float = 0.1,
            attention_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )

        self.attention_W = nn.Linear(attention_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

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

        # Feature extraction
        x = x.view(-1, D)  # [B*P, D]
        h = self.feature_extractor(x)  # [B*P, hidden_dim]
        h = h.view(B, P, -1)  # [B, P, hidden_dim]

        # Gated attention
        A_V = self.attention_V(h)  # [B, P, attention_dim]
        A_U = self.attention_U(h)  # [B, P, attention_dim]
        A = self.attention_W(A_V * A_U)  # [B, P, 1]
        A = A.squeeze(-1)  # [B, P]
        A = F.softmax(A, dim=1)  # [B, P]

        if return_attention:
            self._attention_weights = A.detach()

        # Weighted aggregation
        M = torch.bmm(A.unsqueeze(1), h)  # [B, 1, hidden_dim]
        M = M.squeeze(1)  # [B, hidden_dim]

        # Classification
        logits = self.classifier(M)  # [B, n_classes]

        return logits

    def get_attention_outputs(self):
        """Return stored attention weights."""
        return {'patch_attention': self._attention_weights}


class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning.

    Reference:
        Shao et al. "TransMIL: Transformer based Correlated MIL for WSI Classification" (NeurIPS 2021)

    Architecture:
        patches -> projection -> position_encoding -> transformer -> class_token -> classifier
    """

    def __init__(
            self,
            input_dim: int = 1536,
            hidden_dim: int = 256,
            n_classes: int = 4,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            max_patches: int = 10000
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_patches = max_patches

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Position encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches + 1, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm before classifier
        self.norm = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

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
        x = data.wsi_features
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, P, D]

        B, P, D = x.shape

        # Subsample if too many patches
        if P > self.max_patches:
            idx = torch.randperm(P)[:self.max_patches]
            x = x[:, idx, :]
            P = self.max_patches

        # Project to hidden dimension
        x = self.proj(x)  # [B, P, hidden_dim]

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, P+1, hidden_dim]

        # Add position encoding
        x = x + self.pos_embed[:, :P + 1, :]

        # Transformer
        x = self.transformer(x)  # [B, P+1, hidden_dim]

        # Extract class token
        cls_output = x[:, 0]  # [B, hidden_dim]
        cls_output = self.norm(cls_output)

        # Classification
        logits = self.classifier(cls_output)  # [B, n_classes]

        return logits

    def get_attention_outputs(self):
        """Return stored attention weights (if implemented)."""
        return {'transformer_attention': self._attention_weights}


class DSMIL(nn.Module):
    """
    Dual-Stream Multiple Instance Learning (simplified version).

    Reference:
        Li et al. "Dual-stream MIL Network for Histopathology Image Analysis" (CVPR 2021)

    Uses both instance-level and bag-level classification with attention.
    """

    def __init__(
            self,
            input_dim: int = 1536,
            hidden_dim: int = 256,
            n_classes: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()

        # Instance classifier (critical instance detection)
        self.instance_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Bag classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        self._attention_weights = None
        self._instance_scores = None

    def forward(self, data, return_attention: bool = False):
        """Forward pass combining instance and bag predictions."""
        x = data.wsi_features
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, P, D = x.shape

        # Instance predictions
        x_flat = x.view(-1, D)
        instance_logits = self.instance_classifier(x_flat)  # [B*P, n_classes]
        instance_logits = instance_logits.view(B, P, -1)  # [B, P, n_classes]

        # Find critical instances (top-k attention based on max class score)
        max_scores, _ = instance_logits.max(dim=-1)  # [B, P]

        # Encode features
        h = self.encoder(x_flat).view(B, P, -1)  # [B, P, hidden_dim]

        # Attention-weighted aggregation
        A = self.attention(h).squeeze(-1)  # [B, P]
        A = F.softmax(A, dim=1)

        if return_attention:
            self._attention_weights = A.detach()
            self._instance_scores = instance_logits.detach()

        # Bag representation
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)  # [B, hidden_dim]

        # Bag prediction
        bag_logits = self.bag_classifier(M)  # [B, n_classes]

        # Combine: use critical instance's prediction + bag prediction
        critical_idx = max_scores.argmax(dim=1)  # [B]
        critical_logits = instance_logits[torch.arange(B), critical_idx]  # [B, n_classes]

        # Average the two streams
        logits = 0.5 * bag_logits + 0.5 * critical_logits

        return logits

    def get_attention_outputs(self):
        return {
            'patch_attention': self._attention_weights,
            'instance_scores': self._instance_scores
        }