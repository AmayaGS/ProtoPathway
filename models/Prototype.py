import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoMIL_V0(nn.Module):
    """
    Smallest working prototype-MIL model.
    """
    def __init__(self, input_dim: int, num_prototypes: int = 64, tau: float = 10.0,
                 num_classes: int = 2, init_centroids: torch.Tensor | None = None):

        super().__init__()

        if init_centroids is None:
            self.proto = nn.Parameter(torch.randn(num_prototypes, input_dim))
        else:
            self.proto = nn.Parameter(init_centroids)          # k-means seeds

        self.tau = tau                                         # softmax temp
        self.classifier = nn.Linear(num_prototypes, num_classes)

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

        # 3) bag-level histogram
        h = alpha.mean(dim=0)                                  # [N]

        # 4) classifier
        out = self.classifier(h)

        return out.unsqueeze(0), sim