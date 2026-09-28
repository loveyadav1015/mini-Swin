"""Tests for PatchMerging."""

import torch
from model.swin_transformer import PatchMerging


def test_patch_merging_stage0():
    """(2, 3136, 96) → (2, 784, 192)  [56×56 → 28×28, C doubles]."""
    pm = PatchMerging(input_resolution=(56, 56), dim=96)
    x = torch.randn(2, 3136, 96)
    out = pm(x)
    assert out.shape == (2, 784, 192)


def test_patch_merging_stage1():
    """(2, 784, 192) → (2, 196, 384)  [28×28 → 14×14]."""
    pm = PatchMerging(input_resolution=(28, 28), dim=192)
    x = torch.randn(2, 784, 192)
    out = pm(x)
    assert out.shape == (2, 196, 384)


def test_patch_merging_stage2():
    """(2, 196, 384) → (2, 49, 768)  [14×14 → 7×7]."""
    pm = PatchMerging(input_resolution=(14, 14), dim=384)
    x = torch.randn(2, 196, 384)
    out = pm(x)
    assert out.shape == (2, 49, 768)


def test_patch_merging_channel_doubling():
    """Output channels are exactly 2× input channels."""
    for dim in [96, 128, 192]:
        pm = PatchMerging(input_resolution=(28, 28), dim=dim)
        x = torch.randn(1, 784, dim)
        out = pm(x)
        assert out.shape[-1] == 2 * dim
