"""Tests for WindowAttention."""

import torch
from model.swin_transformer import WindowAttention


def test_window_attention_no_mask():
    """W-MSA: (8, 49, 96), num_heads=3 → same shape out."""
    attn = WindowAttention(dim=96, window_size=7, num_heads=3)
    x = torch.randn(8, 49, 96)
    out = attn(x, mask=None)
    assert out.shape == (8, 49, 96)


def test_window_attention_with_mask():
    """SW-MSA: same shape, different values when mask is applied."""
    attn = WindowAttention(dim=96, window_size=7, num_heads=3)
    attn.eval()

    x = torch.randn(8, 49, 96)

    # Create a dummy mask: 8 windows, 49×49
    mask = torch.zeros(8, 49, 49)
    mask[:4, :, 25:] = -100.0  # mask out half the positions for first 4 windows

    out_no_mask = attn(x, mask=None)
    out_with_mask = attn(x, mask=mask)

    assert out_with_mask.shape == (8, 49, 96)
    # Outputs should differ because the mask changes attention patterns
    assert not torch.allclose(out_no_mask, out_with_mask, atol=1e-6)


def test_relative_position_bias_table_shape():
    """Bias table: ((2*ws-1)², num_heads)."""
    attn = WindowAttention(dim=96, window_size=7, num_heads=3)
    expected_size = (2 * 7 - 1) ** 2  # 169
    assert attn.relative_position_bias_table.shape == (expected_size, 3)


def test_relative_position_index_shape():
    """Index: (49, 49) for ws=7."""
    attn = WindowAttention(dim=96, window_size=7, num_heads=3)
    assert attn.relative_position_index.shape == (49, 49)


def test_window_attention_output_dtype():
    """Output dtype matches input dtype."""
    attn = WindowAttention(dim=96, window_size=7, num_heads=3)
    x = torch.randn(8, 49, 96)
    out = attn(x)
    assert out.dtype == x.dtype
