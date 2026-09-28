"""Tests for PatchEmbedding."""

import torch
from model.swin_transformer import PatchEmbedding


def test_patch_embedding_output_shape():
    """(1, 3, 224, 224) → (1, 3136, 96) for Swin-T defaults."""
    pe = PatchEmbedding(image_size=224, patch_size=4, in_channels=3, embed_dim=96)
    x = torch.randn(1, 3, 224, 224)
    out = pe(x)
    assert out.shape == (1, 3136, 96)


def test_patch_embedding_num_patches():
    """num_patches == (224 // 4)² == 3136."""
    pe = PatchEmbedding(image_size=224, patch_size=4, in_channels=3, embed_dim=96)
    assert pe.num_patches == 3136


def test_patch_embedding_patch_grid():
    """patch_grid == (56, 56)."""
    pe = PatchEmbedding(image_size=224, patch_size=4, in_channels=3, embed_dim=96)
    assert pe.patch_grid == (56, 56)


def test_patch_embedding_batch_independence():
    """Different batch sizes produce correct shapes."""
    pe = PatchEmbedding(image_size=224, patch_size=4, in_channels=3, embed_dim=96)
    for bs in [1, 2, 4]:
        x = torch.randn(bs, 3, 224, 224)
        out = pe(x)
        assert out.shape == (bs, 3136, 96)
