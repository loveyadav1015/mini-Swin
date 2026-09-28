"""Tests for window_partition and window_reverse."""

import torch
from model.swin_transformer import window_partition, window_reverse


def test_window_partition_shape():
    """(1, 56, 56, 96) with ws=7 → (64, 7, 7, 96)."""
    x = torch.randn(1, 56, 56, 96)
    windows = window_partition(x, 7)
    assert windows.shape == (64, 7, 7, 96)


def test_window_partition_num_windows():
    """56 / 7 = 8 windows per side → 8*8 = 64 windows per sample."""
    x = torch.randn(2, 56, 56, 96)
    windows = window_partition(x, 7)
    # 2 samples * 64 windows = 128
    assert windows.shape[0] == 128


def test_window_partition_reverse_roundtrip():
    """window_reverse(window_partition(x)) == x."""
    x = torch.randn(1, 56, 56, 96)
    windows = window_partition(x, 7)
    x_back = window_reverse(windows, 7, 56, 56)
    assert x_back.shape == x.shape
    assert torch.allclose(x, x_back)


def test_window_partition_reverse_batched():
    """Round-trip with batch size > 1."""
    x = torch.randn(4, 56, 56, 96)
    windows = window_partition(x, 7)
    x_back = window_reverse(windows, 7, 56, 56)
    assert torch.allclose(x, x_back)
