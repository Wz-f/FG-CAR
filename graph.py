from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


TEXT_INTRA = 0
IMAGE_INTRA = 1
SEMANTIC_ALIGN = 2
ATTRIBUTE_VERIFY = 3
NUM_RELATIONS = 4


def masked_fill_invalid(scores: Tensor, row_mask: Tensor, col_mask: Tensor, value: float = -1e9) -> Tensor:
    """Mask invalid rows and columns in a batched score matrix."""
    valid = row_mask.unsqueeze(-1) & col_mask.unsqueeze(1)
    return scores.masked_fill(~valid, value)


def cosine_matrix(x: Tensor, y: Tensor) -> Tensor:
    """Batched cosine similarity matrix between x=[B,N,D] and y=[B,M,D]."""
    x = F.normalize(x, dim=-1, eps=1e-8)
    y = F.normalize(y, dim=-1, eps=1e-8)
    return torch.matmul(x, y.transpose(-1, -2))


def bidirectional_topk_adjacency(sim: Tensor, text_mask: Tensor, image_mask: Tensor, k: int) -> Tensor:
    """
    Build sparse cross-modal adjacency using bidirectional Top-K matching.

    sim: [B, Nt, Nv]
    return: [B, Nt, Nv], weighted by cosine similarity.
    """
    bsz, nt, nv = sim.shape
    if nt == 0 or nv == 0:
        return sim.new_zeros(sim.shape)

    scores = masked_fill_invalid(sim, text_mask, image_mask)
    k_t = min(k, nv)
    k_v = min(k, nt)

    top_img = torch.topk(scores, k=k_t, dim=2).indices
    top_text = torch.topk(scores.transpose(1, 2), k=k_v, dim=2).indices

    mask = torch.zeros_like(sim, dtype=torch.bool)
    mask.scatter_(2, top_img, True)

    reverse_mask = torch.zeros_like(sim.transpose(1, 2), dtype=torch.bool)
    reverse_mask.scatter_(2, top_text, True)
    mask = mask | reverse_mask.transpose(1, 2)

    valid = text_mask.unsqueeze(-1) & image_mask.unsqueeze(1)
    mask = mask & valid
    return torch.where(mask, sim, sim.new_zeros(sim.shape))


def topk_attribute_adjacency(attr_sim: Tensor, text_mask: Tensor, image_mask: Tensor, k: int) -> Tensor:
    """
    Build attribute-verification adjacency from textual attribute descriptors
    to the top-K' most relevant visual regions.
    """
    if attr_sim.size(1) == 0 or attr_sim.size(2) == 0:
        return attr_sim.new_zeros(attr_sim.shape)

    scores = masked_fill_invalid(attr_sim, text_mask, image_mask)
    k_attr = min(k, attr_sim.size(2))
    top_img = torch.topk(scores, k=k_attr, dim=2).indices

    mask = torch.zeros_like(attr_sim, dtype=torch.bool)
    mask.scatter_(2, top_img, True)
    valid = text_mask.unsqueeze(-1) & image_mask.unsqueeze(1)
    mask = mask & valid
    return torch.where(mask, attr_sim, attr_sim.new_zeros(attr_sim.shape))


def default_text_adj(text_mask: Tensor) -> Tensor:
    """Fallback textual structural support: self-loops only."""
    bsz, nt = text_mask.shape
    eye = torch.eye(nt, device=text_mask.device, dtype=torch.float32).unsqueeze(0).repeat(bsz, 1, 1)
    valid = text_mask.unsqueeze(-1) & text_mask.unsqueeze(1)
    return eye * valid.float()


def image_adj_from_iou(image_iou: Tensor | None, image_mask: Tensor, threshold: float) -> Tensor:
    """Build visual structural support from IoU; fallback is self-loops only."""
    bsz, nv = image_mask.shape
    eye = torch.eye(nv, device=image_mask.device, dtype=torch.float32).unsqueeze(0).repeat(bsz, 1, 1)
    valid = image_mask.unsqueeze(-1) & image_mask.unsqueeze(1)

    if image_iou is None:
        return eye * valid.float()

    adj = (image_iou >= threshold).float()
    adj = torch.maximum(adj, eye)
    return adj * valid.float()


def pack_relation_adjacency(
    text_nodes: Tensor,
    image_nodes: Tensor,
    text_mask: Tensor,
    image_mask: Tensor,
    attr_nodes: Tensor | None = None,
    text_adj: Tensor | None = None,
    image_iou: Tensor | None = None,
    top_k: int = 5,
    attr_top_k: int = 2,
    iou_threshold: float = 0.5,
    use_attribute_edges: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Construct relation-specific dense adjacency matrices.

    Returns:
        node_features: [B, N, D]
        node_mask: [B, N]
        rel_adj: [B, 4, N, N]
    """
    bsz, nt, dim = text_nodes.shape
    nv = image_nodes.size(1)
    device = text_nodes.device
    n_all = nt + nv

    node_features = torch.cat([text_nodes, image_nodes], dim=1)
    node_mask = torch.cat([text_mask, image_mask], dim=1)

    rel_adj = text_nodes.new_zeros(bsz, NUM_RELATIONS, n_all, n_all)

    # 1. Textual intra-modal structural support
    if text_adj is None:
        text_adj = default_text_adj(text_mask)
    else:
        valid_text = text_mask.unsqueeze(-1) & text_mask.unsqueeze(1)
        eye = torch.eye(nt, device=device, dtype=text_adj.dtype).unsqueeze(0)
        text_adj = torch.maximum(text_adj.float(), eye.repeat(bsz, 1, 1)) * valid_text.float()
    rel_adj[:, TEXT_INTRA, :nt, :nt] = text_adj

    # 2. Visual intra-modal structural support
    image_adj = image_adj_from_iou(image_iou, image_mask, iou_threshold)
    rel_adj[:, IMAGE_INTRA, nt:, nt:] = image_adj

    # 3. Cross-modal semantic alignment
    sim = cosine_matrix(text_nodes, image_nodes)
    a_cross = bidirectional_topk_adjacency(sim, text_mask, image_mask, top_k)
    rel_adj[:, SEMANTIC_ALIGN, :nt, nt:] = a_cross
    rel_adj[:, SEMANTIC_ALIGN, nt:, :nt] = a_cross.transpose(1, 2)

    # 4. Attribute verification
    if use_attribute_edges:
        if attr_nodes is None:
            attr_nodes = text_nodes
        attr_sim = cosine_matrix(attr_nodes, image_nodes)
        a_attr = topk_attribute_adjacency(attr_sim, text_mask, image_mask, attr_top_k)
        rel_adj[:, ATTRIBUTE_VERIFY, :nt, nt:] = a_attr
        rel_adj[:, ATTRIBUTE_VERIFY, nt:, :nt] = a_attr.transpose(1, 2)

    return node_features, node_mask, rel_adj
