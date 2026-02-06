"""
WSI Encoder using Prototype-based Multiple Instance Learning.

Patches are assigned to learned prototypes via soft attention.
Each prototype aggregates its assigned patches into a token embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMIL(nn.Module):
    """
    Prototype-based Multiple Instance Learning for WSI encoding.

    Architecture:
    1. Project patches and prototypes to common embedding space
    2. Compute patch-prototype similarity (cosine)
    3. Soft assignment with temperature-scaled softmax
    4. Learnable gates weight prototype contributions
    5. Aggregate into prototype tokens and slide-level embedding

    The prototypes can be initialized from k-means centroids or randomly.
    """

    def __init__(
            self,
            input_dim,
            hidden_dim=256,
            num_prototypes=64,
            tau=10.0,
            init_centroids=None
    ):
        """
        Initialize the prototype MIL model.

        Args:
            input_dim: Input patch feature dimension (e.g., 1536 for UNI2-h)
            hidden_dim: Projection dimension for prototypes and patches
            num_prototypes: Number of learnable prototypes
            tau: Temperature for softmax attention
            init_centroids: Optional tensor [num_prototypes, input_dim] for initialization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.tau = tau

        # Learnable prototypes
        if init_centroids is not None:
            assert init_centroids.shape == (num_prototypes, input_dim), \
                f"Expected centroids shape ({num_prototypes}, {input_dim}), got {init_centroids.shape}"
            self.proto = nn.Parameter(init_centroids.clone())
        else:
            self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))

        # Dimension reduction (shared for patches and prototypes)
        self.dim_reducer = nn.Linear(input_dim, hidden_dim)

        # Learnable gates for prototype weighting (softplus ensures non-negative)
        self.logit_g = nn.Parameter(torch.zeros(num_prototypes))

        # Storage for visualization
        self.patch_assignments = None

    def forward(self, x, return_assignments=False):
        """
        Forward pass through prototype MIL.

        Args:
            x: Patch features [num_patches, input_dim] (single slide)
            return_assignments: If True, store patch-prototype assignments

        Returns:
            bag_embedding: [hidden_dim] slide-level embedding
            proto_tokens: [num_prototypes, hidden_dim] individual prototype embeddings
        """
        # Add batch dimension for consistency
        x = x.unsqueeze(0)  # [1, num_patches, input_dim]
        B, P, D = x.shape
        N = self.num_prototypes

        # Project prototypes and patches to common space
        proto_reduced = self.dim_reducer(self.proto)  # [N, hidden_dim]

        x_flat = x.view(-1, D)  # [P, input_dim]
        x_reduced = self.dim_reducer(x_flat)  # [P, hidden_dim]
        x_reduced = x_reduced.view(B, P, -1)  # [1, P, hidden_dim]

        # L2 normalize for cosine similarity
        proto_norm = F.normalize(proto_reduced, dim=1)  # [N, hidden_dim]
        x_norm = F.normalize(x_reduced, dim=2)  # [1, P, hidden_dim]

        # Cosine similarity
        similarity = torch.einsum("bpd,nd->bpn", x_norm, proto_norm)  # [1, P, N]

        # Soft assignment with temperature
        alpha = F.softmax(self.tau * similarity, dim=2)  # [1, P, N]

        # Apply learnable gates
        gates = F.softplus(self.logit_g)  # [N], non-negative
        alpha_gated = alpha * gates  # [1, P, N]

        # Prototype-wise pooling
        # proto_token[n] = sum_p(alpha_gated[p,n] * x_reduced[p]) / sum_p(alpha_gated[p,n])
        numerator = torch.einsum("bpn,bpd->bnd", alpha_gated, x_reduced)  # [1, N, hidden_dim]
        denominator = alpha_gated.sum(dim=1, keepdim=False).clamp(min=1e-6)  # [1, N]
        proto_tokens = numerator / denominator.unsqueeze(2)  # [1, N, hidden_dim]

        # Weighted aggregation for slide-level embedding
        gate_weights = gates / gates.sum()  # Normalize gates
        proto_weighted = proto_tokens * gates.view(1, N, 1)
        # bag_embedding = proto_weighted.sum(dim=1)  # [1, hidden_dim]
        bag_embedding = proto_weighted.mean(dim=1)

        # Remove batch dimension
        bag_embedding = bag_embedding.squeeze(0)  # [hidden_dim]
        proto_tokens = proto_weighted.squeeze(0)  # [N, hidden_dim]

        # Store assignments for visualization
        if return_assignments:
            hard_assignments = torch.argmax(alpha.squeeze(0), dim=1)  # [P]
            self.patch_assignments = {
                'soft_assignments': alpha.squeeze(0).detach(),  # [P, N]
                'hard_assignments': hard_assignments.detach(),  # [P]
                'similarities': similarity.squeeze(0).detach(),  # [P, N]
                'gate_weights': gate_weights.detach()  # [N]
            }

        return bag_embedding, proto_tokens

    def get_prototype_assignments(self):
        """Get stored patch-prototype assignments."""
        return self.patch_assignments

    def get_gate_weights(self):
        """Get normalized gate weights for each prototype."""
        gates = F.softplus(self.logit_g)
        return (gates / gates.sum()).detach()


class WSIEncoderUnimodal(nn.Module):
    """
    Standalone WSI encoder for unimodal prediction.

    Wraps PrototypeMIL and adds a classification/survival head.
    """

    def __init__(
            self,
            input_dim,
            num_classes,
            hidden_dim=256,
            num_prototypes=64,
            tau=10.0,
            init_centroids=None
    ):
        super().__init__()

        self.encoder = PrototypeMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_prototypes=num_prototypes,
            tau=tau,
            init_centroids=init_centroids
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_assignments=False):
        """
        Forward pass for unimodal prediction.

        Returns:
            logits: [num_classes] prediction logits
        """
        bag_embedding, _ = self.encoder(x, return_assignments=return_assignments)
        logits = self.classifier(bag_embedding.unsqueeze(0))  # [1, num_classes]
        return logits