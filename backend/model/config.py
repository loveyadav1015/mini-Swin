"""Swin Transformer configuration dataclass and preset factories."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SwinConfig:
    """Hyperparameters for the Swin Transformer architecture.

    All numerical defaults match the Swin-Tiny variant described in
    "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    (Liu et al., 2021).
    """

    # Input
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 4  # 4×4 pixels per patch

    # Architecture
    embed_dim: int = 96  # C — base channel count
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0

    # Regularisation
    drop_path_rate: float = 0.2
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0

    # Head
    num_classes: int = 1000

    # Derived (do not set manually)
    @property
    def num_stages(self) -> int:
        return len(self.depths)


# ── Preset factory functions ──────────────────────────────────────────────────


def swin_tiny() -> SwinConfig:
    """Swin-T: ~28M params."""
    return SwinConfig(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])


def swin_small() -> SwinConfig:
    """Swin-S: ~50M params."""
    return SwinConfig(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])


def swin_base() -> SwinConfig:
    """Swin-B: ~88M params."""
    return SwinConfig(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])


def swin_large() -> SwinConfig:
    """Swin-L: ~197M params."""
    return SwinConfig(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
