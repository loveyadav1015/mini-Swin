"""Swin Transformer architecture — pure PyTorch implementation.

Implements every building block described in "Swin Transformer: Hierarchical
Vision Transformer using Shifted Windows" (Liu et al., ICCV 2021).

Classes / functions (in dependency order):
    DropPath                – stochastic depth regularisation
    PatchEmbedding          – image → patch tokens via strided Conv2d
    window_partition        – (B,H,W,C) → windows
    window_reverse          – windows → (B,H,W,C)
    WindowAttention         – multi-head self-attention inside a window
    SwinTransformerBlock    – one W-MSA or SW-MSA block
    PatchMerging            – spatial ↓2×, channels ×2
    SwinStage               – N blocks + optional PatchMerging
    SwinTransformer         – full model (4 stages + head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .config import SwinConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Stochastic depth — drops the entire sample with probability *drop_prob*
    during training and scales the surviving samples to preserve expected value.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (B, 1, 1, …) — one random scalar per sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int,
                 drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ── Patch Embedding ───────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly embed them.

    (B, C_in, H, W) → (B, num_patches, embed_dim)
    """

    def __init__(self, image_size: int = 224, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)           # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ── Window utilities ──────────────────────────────────────────────────────────


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition a feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: window edge length

    Returns:
        (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    # (B, nH, ws, nW, ws, C) → (B, nH, nW, ws, ws, C) → (B*nH*nW, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """Reverse of :func:`window_partition`.

    Args:
        windows: (num_windows * B, window_size, window_size, C)
        window_size: window edge length
        H, W: original spatial dimensions

    Returns:
        (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ── Window Attention ──────────────────────────────────────────────────────────


class WindowAttention(nn.Module):
    """Multi-head self-attention computed within each local window.

    Supports an optional attention mask for shifted-window (SW-MSA) mode.
    Includes learnable relative position bias.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Build the relative position index (constant — registered as buffer)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = torch.flatten(coords, 1)  # (2, ws*ws)

        # Pairwise relative coordinates
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()      # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C)  where N = window_size²
            mask: (num_windows, N, N) or None.  Added to attention logits.

        Returns:
            (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply cyclic attention mask (shifted windows)
        if mask is not None:
            nW = mask.shape[0]  # number of windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # broadcast
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── Swin Transformer Block ───────────────────────────────────────────────────


class SwinTransformerBlock(nn.Module):
    """One Swin Transformer block — either W-MSA (shift_size=0) or
    SW-MSA (shift_size=window_size//2).

    Pre-norm architecture:
        x → LN → WindowAttn → residual → LN → MLP → DropPath → residual
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop_path: float = 0.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 input_resolution: Tuple[int, int] = (56, 56)) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.input_resolution = input_resolution

        # If the feature map is smaller than the window, disable windowing
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
        )

        # Build cyclic attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)

        Returns:
            (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows back
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        # Residual + DropPath
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Patch Merging ─────────────────────────────────────────────────────────────


class PatchMerging(nn.Module):
    """Downsample spatial resolution by 2× and double the channel count.

    Concatenates the four 2×2 neighbours, applies LayerNorm, then a linear
    projection from 4C → 2C.

    (B, H*W, C)  →  (B, H/2 * W/2, 2C)
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ── Swin Stage ────────────────────────────────────────────────────────────────


class SwinStage(nn.Module):
    """A sequence of Swin Transformer blocks for one stage, alternating
    W-MSA (even index) and SW-MSA (odd index), with an optional PatchMerging
    downsample at the end.
    """

    def __init__(self, dim: int, input_resolution: Tuple[int, int],
                 depth: int, num_heads: int, window_size: int,
                 drop_path_rates: List[float],
                 downsample: bool = True,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[i],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                input_resolution=input_resolution,
            )
            for i in range(depth)
        ])

        if downsample:
            self.downsample: Optional[PatchMerging] = PatchMerging(
                input_resolution, dim=dim
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ── Full Model ────────────────────────────────────────────────────────────────


class SwinTransformer(nn.Module):
    """Swin Transformer — hierarchical vision transformer using shifted windows.

    Architecture::

        PatchEmbedding
            │
            ├── Stage 0 (depth=d₀, dim=C,   res=H/4 × W/4)   + PatchMerging
            ├── Stage 1 (depth=d₁, dim=2C,  res=H/8 × W/8)   + PatchMerging
            ├── Stage 2 (depth=d₂, dim=4C,  res=H/16×W/16)   + PatchMerging
            └── Stage 3 (depth=d₃, dim=8C,  res=H/32×W/32)   — no PatchMerging
                 │
            LayerNorm → AdaptiveAvgPool1d → Flatten → Linear(8C, num_classes)
    """

    def __init__(self, config: SwinConfig) -> None:
        super().__init__()
        self.config = config
        self.num_stages = config.num_stages

        # ── Patch embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        # ── Stochastic depth schedule (linear) ──────────────────────────────
        total_blocks = sum(config.depths)
        dp_rates = [
            r.item()
            for r in torch.linspace(0, config.drop_path_rate, total_blocks)
        ]

        # ── Build stages ─────────────────────────────────────────────────────
        patches_resolution = (
            config.image_size // config.patch_size,
            config.image_size // config.patch_size,
        )

        self.stages = nn.ModuleList()
        for i_stage in range(self.num_stages):
            dim = int(config.embed_dim * (2 ** i_stage))
            resolution = (
                patches_resolution[0] // (2 ** i_stage),
                patches_resolution[1] // (2 ** i_stage),
            )
            depth = config.depths[i_stage]
            # Slice the correct drop-path rates for this stage
            dp_start = sum(config.depths[:i_stage])
            dp_end = dp_start + depth

            stage = SwinStage(
                dim=dim,
                input_resolution=resolution,
                depth=depth,
                num_heads=config.num_heads[i_stage],
                window_size=config.window_size,
                drop_path_rates=dp_rates[dp_start:dp_end],
                downsample=(i_stage < self.num_stages - 1),
                mlp_ratio=config.mlp_ratio,
                attn_drop=config.attn_drop_rate,
                proj_drop=config.proj_drop_rate,
            )
            self.stages.append(stage)

        # ── Classification head ──────────────────────────────────────────────
        final_dim = int(config.embed_dim * (2 ** (self.num_stages - 1)))
        self.norm = nn.LayerNorm(final_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(final_dim, config.num_classes)

        # ── Weight initialisation ────────────────────────────────────────────
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  input image batch

        Returns:
            (B, num_classes)  classification logits
        """
        x = self.patch_embed(x)          # (B, num_patches, C)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)                 # (B, L', 8C)
        x = self.avgpool(x.transpose(1, 2))  # (B, 8C, 1)
        x = torch.flatten(x, 1)         # (B, 8C)
        x = self.head(x)                # (B, num_classes)
        return x

    def num_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())
