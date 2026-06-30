from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graph import NUM_RELATIONS


class RelationAwareGraphAttentionLayer(nn.Module):

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.3, num_relations: int = NUM_RELATIONS):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_relations = num_relations

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_relations)])
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        bsz, n_nodes, _ = x.shape
        return x.view(bsz, n_nodes, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, rel_adj: Tensor, node_mask: Tensor) -> Tensor:
        """
        x: [B, N, D]
        rel_adj: [B, R, N, N], adjacency from source j to target i is rel_adj[..., i, j]
        node_mask: [B, N]
        """
        bsz, n_nodes, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))

        valid_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # [B,N,N]
        rel_outputs = []

        for r in range(self.num_relations):
            adj = rel_adj[:, r]  # [B,N,N]
            edge_mask = (adj != 0) & valid_pair
            if not edge_mask.any():
                rel_outputs.append(torch.zeros_like(x))
                continue

            v = self._reshape_heads(self.v_proj[r](x))  # [B,H,N,Dh]
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # Use edge strength as an additive attention bias.
            edge_weight = ((adj + 1.0) / 2.0).clamp(1e-6, 1.0)
            logits = logits + torch.log(edge_weight).unsqueeze(1)

            logits = logits.masked_fill(~edge_mask.unsqueeze(1), -1e9)
            attn = F.softmax(logits, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)  # [B,H,N,Dh]
            out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, self.dim)
            rel_outputs.append(out)

        h = torch.stack(rel_outputs, dim=0).sum(dim=0)
        h = self.out_proj(h)
        h = self.dropout(h)

        out = self.norm(x + h)
        out = out * node_mask.unsqueeze(-1).float()
        return out


class GraphEncoder(nn.Module):
    """Stacked relation-aware graph attention with gated graph pooling."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            RelationAwareGraphAttentionLayer(dim=dim, heads=heads, dropout=dropout)
            for _ in range(layers)
        ])
        self.pool_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, node_features: Tensor, rel_adj: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
        h = node_features
        for layer in self.layers:
            h = layer(h, rel_adj, node_mask)

        gate_logits = self.pool_gate(h).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~node_mask, -1e9)
        gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
        graph_emb = (gate * h).sum(dim=1)
        return h, graph_emb
