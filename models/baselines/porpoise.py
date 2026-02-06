import torch
import torch.nn as nn
import torch.nn.functional as F


def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Network block."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


def MLP_Block(dim1, dim2, dropout=0.25):
    """Standard MLP block."""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.Dropout(p=dropout, inplace=False)
    )


class Attn_Net_Gated(nn.Module):
    """Gated Attention Network (from original PORPOISE)."""

    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class BilinearFusion(nn.Module):
    """
    Bilinear fusion with gated multimodal units (from original PORPOISE).
    """

    def __init__(
            self,
            dim1=256,
            dim2=256,
            scale_dim1=8,
            scale_dim2=8,
            gate1=True,
            gate2=True,
            skip=True,
            mmhid=256,
            dropout=0.25
    ):
        super().__init__()
        self.skip = skip
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_out = dim1 // scale_dim1
        dim2_out = dim2 // scale_dim2
        skip_dim = dim1 + dim2 if skip else 0

        # Gated unit for modality 1 (WSI)
        self.linear_h1 = nn.Sequential(nn.Linear(dim1, dim1_out), nn.ReLU())
        self.linear_z1 = nn.Sequential(nn.Linear(dim1 + dim2, dim1_out))
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1_out, dim1_out),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Gated unit for modality 2 (Omic)
        self.linear_h2 = nn.Sequential(nn.Linear(dim2, dim2_out), nn.ReLU())
        self.linear_z2 = nn.Sequential(nn.Linear(dim1 + dim2, dim2_out))
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2_out, dim2_out),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Bilinear fusion
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1_out + 1) * (dim2_out + 1), 256),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256 + skip_dim, mmhid),
            nn.ReLU()
        )

    def forward(self, vec1, vec2):
        # Gated multimodal unit for WSI
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            o1 = self.linear_o1(self.linear_h1(vec1))

        # Gated multimodal unit for Omic
        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
        else:
            o2 = self.linear_o2(self.linear_h2(vec2))

        # Bilinear fusion via outer product
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=o1.device)), dim=1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=o2.device)), dim=1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)

        if self.skip:
            out = torch.cat((out, vec1, vec2), dim=1)

        out = self.encoder2(out)
        return out


class PORPOISE(nn.Module):
    """
    PORPOISE (PorpoiseMMF): Multimodal fusion for survival prediction.

    Architecture:
        WSI: patches -> gated attention -> pooled embedding (256)
        Omic: flattened genes -> SNN -> embedding (256)
        Fusion: bilinear or concat -> classifier

    Note: Does NOT use pathway structure - serves as baseline to show
    that pathway organization improves performance.
    """

    def __init__(
            self,
            num_genes=int,
            num_features: int = 1536,
            fusion: str = 'bilinear',
            hidden_dim: int = 256,
            n_classes: int = 4,
            dropout: float = 0.25,
            drop_input: float = 0.10,
            gate_wsi: bool = True,
            gate_omic: bool = True,
            skip: bool = True,
            use_mlp: bool = False,
    ):
        super().__init__()

        self.fusion_type = fusion
        self.num_genes = num_genes
        self.n_classes = n_classes

        # --- WSI Branch ---
        # Input dropout + projection + gated attention
        wsi_layers = []
        if drop_input > 0:
            wsi_layers.append(nn.Dropout(drop_input))
        wsi_layers.extend([
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        wsi_layers.append(Attn_Net_Gated(L=512, D=hidden_dim, dropout=dropout, n_classes=1))
        self.attention_net = nn.Sequential(*wsi_layers[:-1])
        self.attention_head = wsi_layers[-1]

        self.rho = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Omic Branch ---
        Block = MLP_Block if use_mlp else SNN_Block
        self.fc_omic = nn.Sequential(
            Block(num_genes, hidden_dim),
            Block(hidden_dim, hidden_dim, dropout=0.25),
        )

        # --- Fusion ---
        if fusion == 'bilinear':
            self.mm = BilinearFusion(
                dim1=hidden_dim,
                dim2=hidden_dim,
                scale_dim1=8,
                scale_dim2=8,
                gate1=gate_wsi,
                gate2=gate_omic,
                skip=skip,
                mmhid=hidden_dim,
                dropout=dropout
            )
        elif fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # --- Classifier ---
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # Storage for attention
        self._wsi_attention = None

    def forward(self, data, return_attention: bool = False):
        """
        Forward pass.

        Args:
            data: PyG Data with:
                - x: [num_nodes, 1] gene expression (first num_genes rows)
                - wsi_features: [num_patches, num_features]

        Returns:
            logits: [1, n_classes]
        """
        # --- WSI Branch ---
        wsi_features = data.wsi_features
        if wsi_features.dim() == 3:
            wsi_features = wsi_features.squeeze(0)

        h_wsi = self.attention_net(wsi_features)  # [N, 512]
        A, h_wsi = self.attention_head(h_wsi)  # A: [N, 1], h_wsi: [N, 512]

        A = torch.transpose(A, 1, 0)  # [1, N]
        A = F.softmax(A, dim=1)

        if return_attention:
            self._wsi_attention = A.detach()

        h_wsi = torch.mm(A, h_wsi)  # [1, 512]
        h_wsi = self.rho(h_wsi)  # [1, hidden_dim]

        # --- Omic Branch ---
        num_genes = self.num_genes
        if torch.is_tensor(num_genes):
            num_genes = int(num_genes.item())

        gene_x = data.x[:num_genes].squeeze(-1)  # [num_genes]
        h_omic = self.fc_omic(gene_x.unsqueeze(0))  # [1, hidden_dim]

        # --- Fusion ---
        if self.fusion_type == 'bilinear':
            h_mm = self.mm(h_wsi, h_omic)  # [1, hidden_dim]
        else:  # concat
            h_mm = self.mm(torch.cat([h_wsi, h_omic], dim=1))

        # --- Classifier ---
        logits = self.classifier(h_mm)  # [1, n_classes]

        return logits

    def get_attention_outputs(self):
        return {'wsi_attention': self._wsi_attention}
