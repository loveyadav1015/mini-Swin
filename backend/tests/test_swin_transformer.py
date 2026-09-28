"""End-to-end forward pass tests for all Swin Transformer variants."""

import torch
from model import SwinTransformer, swin_tiny, swin_small, swin_base, swin_large


def test_swin_tiny_forward():
    """Swin-T: (1, 3, 224, 224) → (1, 1000)."""
    model = SwinTransformer(swin_tiny())
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_swin_small_forward():
    """Swin-S: (1, 3, 224, 224) → (1, 1000)."""
    model = SwinTransformer(swin_small())
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_swin_base_forward():
    """Swin-B: (1, 3, 224, 224) → (1, 1000)."""
    model = SwinTransformer(swin_base())
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_swin_large_forward():
    """Swin-L: (1, 3, 224, 224) → (1, 1000)."""
    model = SwinTransformer(swin_large())
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000)


def test_swin_tiny_batch():
    """Swin-T with batch_size > 1."""
    model = SwinTransformer(swin_tiny())
    model.eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1000)
