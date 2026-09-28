"""Tests for SwinTransformerBlock."""

import torch
from model.swin_transformer import SwinTransformerBlock


def test_wmsa_block_shape():
    """W-MSA block: (2, 3136, 96) → (2, 3136, 96)."""
    block = SwinTransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=0,
        input_resolution=(56, 56),
    )
    x = torch.randn(2, 3136, 96)
    out = block(x)
    assert out.shape == (2, 3136, 96)


def test_swmsa_block_shape():
    """SW-MSA block: (2, 3136, 96) → (2, 3136, 96)."""
    block = SwinTransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=3,
        input_resolution=(56, 56),
    )
    x = torch.randn(2, 3136, 96)
    out = block(x)
    assert out.shape == (2, 3136, 96)


def test_block_output_dtype():
    """Output dtype matches input dtype."""
    block = SwinTransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=0,
        input_resolution=(56, 56),
    )
    x = torch.randn(2, 3136, 96)
    out = block(x)
    assert out.dtype == x.dtype


def test_wmsa_vs_swmsa_differ():
    """W-MSA and SW-MSA blocks produce different outputs."""
    torch.manual_seed(42)
    block_w = SwinTransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=0,
        input_resolution=(56, 56),
    )
    block_sw = SwinTransformerBlock(
        dim=96, num_heads=3, window_size=7, shift_size=3,
        input_resolution=(56, 56),
    )
    block_w.eval()
    block_sw.eval()

    x = torch.randn(1, 3136, 96)
    out_w = block_w(x)
    out_sw = block_sw(x)
    # They should differ because the shifted window has a different attention pattern
    assert not torch.allclose(out_w, out_sw, atol=1e-6)
