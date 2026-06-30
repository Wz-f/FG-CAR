from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .augment import SemanticAwareGraphAugmentor
from .config import FGCARConfig
from .graph import pack_relation_adjacency
from .layers import GraphEncoder
from .losses import GraphContrastiveLoss, GlobalContrastiveLoss
from .modules import ConflictAwareAdaptiveFusion, ProjectionHead


class FGCAR(nn.Module):

    def __init__(self, config: FGCARConfig | None = None):
        super().__init__()
        self.config = config or FGCARConfig()

        cfg = self.config
        self.input_proj = nn.Linear(cfg.d_model, cfg.hidden_dim)

        self.graph_encoder = GraphEncoder(
            dim=cfg.hidden_dim,
            layers=cfg.gat_layers,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
        )

        self.augmentor = SemanticAwareGraphAugmentor(
            rho=cfg.rho,
            use_semantic_augmentation=cfg.use_semantic_augmentation,
        )

        self.text_global_proj = ProjectionHead(cfg.d_model, cfg.proj_dim, cfg.dropout)
        self.image_global_proj = ProjectionHead(cfg.d_model, cfg.proj_dim, cfg.dropout)

        self.caf = ConflictAwareAdaptiveFusion(
            dim=cfg.d_model,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_caf=cfg.use_caf,
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.num_classes),
        )

        self.graph_cl = GraphContrastiveLoss(cfg.temperature)
        self.global_cl = GlobalContrastiveLoss(cfg.temperature)

    @staticmethod
    def _default_mask(x: Tensor) -> Tensor:
        # Valid nodes are assumed to have at least one non-zero feature.
        return x.abs().sum(dim=-1) > 0

    def _encode_graph(
        self,
        text_nodes: Tensor,
        image_nodes: Tensor,
        text_mask: Tensor,
        image_mask: Tensor,
        attr_nodes: Tensor | None,
        text_adj: Tensor | None,
        image_iou: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        node_features, node_mask, rel_adj = pack_relation_adjacency(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=text_mask,
            image_mask=image_mask,
            attr_nodes=attr_nodes,
            text_adj=text_adj,
            image_iou=image_iou,
            top_k=cfg.top_k,
            attr_top_k=cfg.attr_top_k,
            iou_threshold=cfg.iou_threshold,
            use_attribute_edges=cfg.use_attribute_edges,
        )
        node_features = self.input_proj(node_features)
        _, graph_emb = self.graph_encoder(node_features, rel_adj, node_mask)
        return graph_emb, node_features, node_mask, rel_adj

    def forward(
        self,
        text_global: Tensor,
        image_global: Tensor,
        text_nodes: Tensor,
        image_nodes: Tensor,
        attr_nodes: Tensor | None = None,
        text_mask: Tensor | None = None,
        image_mask: Tensor | None = None,
        text_adj: Tensor | None = None,
        image_iou: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            text_global: [B,D]
            image_global: [B,D]
            text_nodes: [B,Nt,D]
            image_nodes: [B,Nv,D]
            attr_nodes: optional [B,Nt,D]
            text_mask: optional [B,Nt]
            image_mask: optional [B,Nv]
            text_adj: optional [B,Nt,Nt]
            image_iou: optional [B,Nv,Nv]
            labels: optional [B], 1=fake, 0=real
        """
        cfg = self.config

        if text_mask is None:
            text_mask = self._default_mask(text_nodes)
        if image_mask is None:
            image_mask = self._default_mask(image_nodes)

        graph_emb, node_features, node_mask, rel_adj = self._encode_graph(
            text_nodes=text_nodes,
            image_nodes=image_nodes,
            text_mask=text_mask,
            image_mask=image_mask,
            attr_nodes=attr_nodes,
            text_adj=text_adj,
            image_iou=image_iou,
        )

        # Augmented graph representation for micro-level graph contrastive learning
        loss_graph = text_global.new_tensor(0.0)
        if self.training and cfg.use_graph_cl:
            rel_adj_aug = self.augmentor(rel_adj)
            _, graph_emb_aug = self.graph_encoder(node_features, rel_adj_aug, node_mask)
            loss_graph = self.graph_cl(graph_emb, graph_emb_aug)

        # Macro-level global semantic alignment
        z_text = self.text_global_proj(text_global)
        z_image = self.image_global_proj(image_global)
        loss_global = text_global.new_tensor(0.0)
        if self.training and cfg.use_global_cl:
            loss_global = self.global_cl(z_text, z_image)

        # CAF and classification
        fused, alpha = self.caf(text_global, image_global, graph_emb)
        logits = self.classifier(fused)
        prob = torch.softmax(logits, dim=-1)

        output: dict[str, Any] = {
            "logits": logits,
            "prob": prob,
            "graph_emb": graph_emb,
            "alpha": alpha,
            "loss_graph": loss_graph,
            "loss_global": loss_global,
        }

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            total_loss = loss_cls
            if cfg.use_graph_cl:
                total_loss = total_loss + cfg.lambda_graph * loss_graph
            if cfg.use_global_cl:
                total_loss = total_loss + cfg.lambda_global * loss_global

            output.update({
                "loss_cls": loss_cls,
                "loss": total_loss,
            })

        return output
