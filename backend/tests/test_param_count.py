"""Tests that parameter counts match the published Swin Transformer paper."""

import torch
from model import SwinTransformer, swin_tiny, swin_small, swin_base, swin_large


def _param_count(config_fn) -> int:
    model = SwinTransformer(config_fn())
    return model.num_parameters()


def test_swin_tiny_params():
    """Swin-T: 28M ± 1M."""
    count = _param_count(swin_tiny)
    assert 27_000_000 <= count <= 29_000_000, f"Swin-T param count: {count:,}"


def test_swin_small_params():
    """Swin-S: 50M ± 1M."""
    count = _param_count(swin_small)
    assert 49_000_000 <= count <= 51_000_000, f"Swin-S param count: {count:,}"


def test_swin_base_params():
    """Swin-B: 88M ± 2M."""
    count = _param_count(swin_base)
    assert 86_000_000 <= count <= 90_000_000, f"Swin-B param count: {count:,}"


def test_swin_large_params():
    """Swin-L: 197M ± 3M."""
    count = _param_count(swin_large)
    assert 194_000_000 <= count <= 200_000_000, f"Swin-L param count: {count:,}"
