import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoMIL_V1(nn.Module):
    """
    Smallest working prototype-MIL model.
    """
    def __init__(self, config, input_dim: int, embedding_dim, num_prototypes: int = 64, tau: float = 10.0,
                 num_classes: int = 2, init_centroids: torch.Tensor | None = None):

        super().__init__()

        self.config = config

        if init_centroids is None:
            self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))
        else:
            self.proto = nn.Parameter(init_centroids)          # k-means seeds

        # Add dimension reduction layer
        self.dim_reducer = nn.Linear(input_dim, embedding_dim)

        # (b) non-negative gates (soft-plus ensures ≥0)
        self.logit_g = nn.Parameter(torch.zeros(num_prototypes))

        self.tau = tau                                         # softmax temp
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        x  – patch-level embeddings for a batch of slides
             shape: [B_slide , P_patch , D]    (NOT [B, D])
        """
        x = x.unsqueeze(0)  # [B, P, D]  (batch size = 1)
        B, P, D = x.shape
        N = self.proto.shape[0]

        # 1) Reduce dimensions of both prototypes and input
        p_reduced = self.dim_reducer(self.proto)  # [N, embedding_dim]

        # Reshape input for batch processing through linear layer
        x_flat = x.view(-1, D)  # [B*P, D]
        x_reduced = self.dim_reducer(x_flat)  # [B*P, embedding_dim]
        x_reduced = x_reduced.view(B, P, -1)  # [B, P, embedding_dim]

        # 2) L2-normalise & cosine similarity
        p = F.normalize(p_reduced, dim=1)  # [N, D]
        x_n = F.normalize(x_reduced, dim=2)  # [B, P, D]
        sim = torch.einsum("bpd,nd->bpn", x_n, p)  # [B, P, N]

        # 3) soft assignment
        alpha = F.softmax(self.tau * sim, dim=2)  # [B, P, N]
        gates = F.softplus(self.logit_g)  # [N]
        alpha_g = alpha * gates  # broadcast to [B, P, N]

        # 4) prototype-wise pooling  →  tokens
        #    µ_{b,n} = Σ_p α̃_{bpn}·x_{bp}  /  Σ_p α̃_{bpn}
        numer = torch.einsum("bpn,bpd->bnd", alpha_g, x_reduced)  # [B, N, D]
        denom = alpha_g.sum(dim=1, keepdim=False).clamp(min=1e-6)  # [B, N]
        proto_tok = numer / denom.unsqueeze(2)  # [B, N, D]

        # 4) slide-level representation for the unimodal baseline
        #    (simple mean over prototype tokens)
        bag_repr = proto_tok.mean(dim=1)  # [B, D]
        logits = self.classifier(bag_repr)  # [B, C]

        if self.config['execution']['mode'] == 'multimodal':
            return bag_repr, proto_tok
        else:
            return logits, sim



class ProtoMIL_V0(nn.Module):
    """
    Smallest working prototype-MIL model.
    """
    def __init__(self, config, input_dim: int, num_prototypes: int = 64, tau: float = 10.0, num_classes: int = 2,
                 init_centroids: torch.Tensor | None = None):

        super().__init__()

        if init_centroids is None:
            self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))
        else:
            self.proto = nn.Parameter(init_centroids)          # k-means seeds

        # (b) non-negative gates (soft-plus ensures ≥0)
        self.logit_g = nn.Parameter(torch.zeros(num_prototypes))

        self.tau = tau                                         # softmax temp
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x – [B, D] patch embeddings from a *single* slide
        """
        # 1) cosine distance → similarity
        p = F.normalize(self.proto, dim=1)                     # [N, D]
        x = F.normalize(x, dim=1)                              # [B, D]
        sim = torch.matmul(x, p.T)                             # [B, N]

        # 2) soft assignment
        alpha = F.softmax(self.tau * sim, dim=1)               # [B, N]
        gates = F.softplus(self.logit_g)  # g ≥ 0
        weighted_alpha = alpha * gates  # ã

        proto_sum = torch.matmul(weighted_alpha, p)
        #max_proto_idx = torch.argmax(sim, dim=1)  # [batch_size]

        # 3) bag-level histogram
        h = proto_sum.mean(dim=0)                                  # [N]

        # 4) classifier
        out = self.classifier(h)

        return out.unsqueeze(0), sim


# class ProtoMIL_V0(nn.Module):
#     """
#     Smallest working prototype-MIL model.
#     """
#     def __init__(self, input_dim: int, num_prototypes: int = 64, tau: float = 10.0,
#                  num_classes: int = 2, init_centroids: torch.Tensor | None = None):
#
#         super().__init__()
#
#         if init_centroids is None:
#             self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))
#         else:
#             self.proto = nn.Parameter(init_centroids)          # k-means seeds
#
#         self.tau = tau                                         # softmax temp
#         self.classifier = nn.Linear(num_prototypes, num_classes)
#
#     def forward(self, x):
#         """
#         x – [B, D] patch embeddings from a *single* slide
#         """
#         # 1) cosine distance → similarity
#         p = F.normalize(self.proto, dim=1)                     # [N, D]
#         x = F.normalize(x, dim=1)                              # [B, D]
#         sim = torch.matmul(x, p.T)                             # [B, N]
#
#         # 2) soft assignment
#         alpha = F.softmax(self.tau * sim, dim=1)               # [B, N]
#
#         # 3) bag-level histogram
#         h = alpha.mean(dim=0)                                  # [N]
#
#         # 4) classifier
#         out = self.classifier(h)
#
#         return out.unsqueeze(0), sim