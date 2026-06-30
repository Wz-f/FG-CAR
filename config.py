from dataclasses import dataclass


@dataclass
class FGCARConfig:
    """
    Paper-related configuration only.

    This config intentionally keeps only the hyperparameters that correspond to
    the FG-CAR method and experimental setting described in the manuscript.
    Dataset paths, preprocessing paths, and file-system settings are excluded.
    """

    # CLIP feature dimension. For common ViT-B CLIP checkpoints this is 512.
    # Change it if the released feature extractor uses another CLIP variant.
    d_model: int = 512

    # Task
    num_classes: int = 2

    # Cross-modal reasoning graph
    top_k: int = 5                 # bidirectional Top-K semantic matching
    attr_top_k: int = 2            # K' for attribute verification edges
    iou_threshold: float = 0.5     # visual structural-support edge threshold
    use_global_node: bool = False  # global nodes are excluded from graph topology

    # Semantic-aware graph augmentation
    rho: float = 0.8               # edge-retention scaling coefficient in Eq. (7)

    # GAT encoder
    hidden_dim: int = 256
    gat_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.3

    # Projection head and contrastive learning
    proj_dim: int = 128
    temperature: float = 0.07
    lambda_graph: float = 0.05     # lambda_1
    lambda_global: float = 0.4     # lambda_2

    # Backbone state stated in the manuscript
    clip_frozen: bool = True

    # Ablation controls. Keep all True for the full model.
    use_attribute_edges: bool = True
    use_semantic_augmentation: bool = True
    use_graph_cl: bool = True
    use_global_cl: bool = True
    use_caf: bool = True
