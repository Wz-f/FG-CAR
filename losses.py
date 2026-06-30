from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_orig: Tensor, h_aug: Tensor) -> Tensor:
        h_orig = F.normalize(h_orig, dim=-1, eps=1e-8)
        h_aug = F.normalize(h_aug, dim=-1, eps=1e-8)
        logits = torch.matmul(h_orig, h_aug.t()) / self.temperature
        labels = torch.arange(h_orig.size(0), device=h_orig.device)
        return F.cross_entropy(logits, labels)


class GlobalContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_text: Tensor, z_image: Tensor) -> Tensor:
        z_text = F.normalize(z_text, dim=-1, eps=1e-8)
        z_image = F.normalize(z_image, dim=-1, eps=1e-8)

        logits = torch.matmul(z_text, z_image.t()) / self.temperature
        labels = torch.arange(z_text.size(0), device=z_text.device)

        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_t2i + loss_i2t)
