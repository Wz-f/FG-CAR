from __future__ import annotations

import torch
from torch import Tensor

from .graph import SEMANTIC_ALIGN, ATTRIBUTE_VERIFY


class SemanticAwareGraphAugmentor(torch.nn.Module):

    def __init__(self, rho: float = 0.8, use_semantic_augmentation: bool = True):
        super().__init__()
        self.rho = rho
        self.use_semantic_augmentation = use_semantic_augmentation

    @staticmethod
    def _normalize_affinity(a: Tensor) -> Tensor:
        # The cosine edge value may be negative. Map valid values to [0,1].
        return ((a + 1.0) / 2.0).clamp(0.0, 1.0)

    def forward(self, rel_adj: Tensor) -> Tensor:
        """
        rel_adj: [B, R, N, N]
        return augmented relation adjacency of the same shape.
        """
        if not self.training:
            return rel_adj

        aug = rel_adj.clone()
        cross_relations = [SEMANTIC_ALIGN, ATTRIBUTE_VERIFY]

        for r in cross_relations:
            edge = aug[:, r]
            valid = edge != 0
            if not valid.any():
                continue

            if self.use_semantic_augmentation:
                prob = self.rho * self._normalize_affinity(edge)
                prob = prob.clamp(0.0, 1.0)
            else:
                prob = torch.full_like(edge, self.rho)

            keep = (torch.rand_like(edge) < prob) & valid
            aug[:, r] = torch.where(keep, edge, torch.zeros_like(edge))

        return aug
