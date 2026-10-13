"""Loss functions for Swin Transformer training.

Includes:
- LabelSmoothingCrossEntropy  — softened CE to prevent overconfident predictions
- mixup_data / cutmix_data    — input-space augmentation
- mixup_criterion             — blended loss for augmented batches
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Args:
        smoothing: Amount of label mass spread to non-target classes (default 0.1).

    Example::

        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = criterion(logits, targets)
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C) raw model outputs (pre-softmax).
            targets: (N,)   integer class indices in [0, C).

        Returns:
            Scalar loss averaged over the batch.
        """
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Hard-label component: NLL at the true class
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Smooth component: uniform over all classes
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ── MixUp ────────────────────────────────────────────────────────────────────


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch.

    Args:
        x:     (N, C, H, W) input images.
        y:     (N,) integer labels.
        alpha: Beta distribution concentration parameter.

    Returns:
        mixed_x: blended images.
        y_a:     original labels.
        y_b:     permuted labels.
        lam:     mixing coefficient (scalar float).
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam: float = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ── CutMix ───────────────────────────────────────────────────────────────────


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch.

    A rectangular patch from a randomly selected image is pasted into each
    image. The mixing coefficient ``lam`` is adjusted to reflect the actual
    patch area ratio.

    Args:
        x:     (N, C, H, W) input images.
        y:     (N,) integer labels.
        alpha: Beta distribution concentration parameter.

    Returns:
        x:   images with patches replaced (in-place on the input tensor).
        y_a: original labels.
        y_b: permuted labels.
        lam: actual area-adjusted mixing coefficient.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam: float = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Recompute lam based on actual patch area
    lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(W * H)

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# ── Combined criterion ────────────────────────────────────────────────────────


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the blended loss for a MixUp/CutMix augmented batch.

    Args:
        criterion: Loss function (e.g. LabelSmoothingCrossEntropy).
        pred:      (N, C) model predictions.
        y_a:       (N,) primary labels.
        y_b:       (N,) secondary (permuted) labels.
        lam:       Mixing coefficient — weight on y_a.

    Returns:
        Scalar blended loss.
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
