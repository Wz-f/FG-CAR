from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ProjectionHead(nn.Module):
    """MLP projection head for global text-image contrastive learning."""

    def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConflictAwareAdaptiveFusion(nn.Module):


    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3, use_caf: bool = True):
        super().__init__()
        self.use_caf = use_caf

        self.text_align = nn.Linear(dim, hidden_dim)
        self.image_align = nn.Linear(dim, hidden_dim)
        self.graph_align = nn.Linear(hidden_dim, hidden_dim)

        self.w_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)

        self.no_caf_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_global: Tensor, image_global: Tensor, graph_emb: Tensor) -> tuple[Tensor, Tensor | None]:
        t = self.text_align(text_global)
        v = self.image_align(image_global)
        g = self.graph_align(graph_emb)

        h_global = self.w_fusion(t + v)
        h_global = self.dropout(h_global)

        if not self.use_caf:
            return self.no_caf_fusion(torch.cat([g, h_global], dim=-1)), None

        discrepancy = torch.cat([torch.abs(t - v), t * v, g], dim=-1)
        alpha = torch.sigmoid(self.gate(torch.tanh(self.context(discrepancy))))

        fused = alpha * g + (1.0 - alpha) * h_global
        return fused, alpha
