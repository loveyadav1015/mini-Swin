"""Learning-rate schedulers for Swin Transformer training."""

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    """Linear warm-up followed by cosine annealing.

    The schedule has two phases:
    1. **Warm-up** (epochs 0 … warmup_epochs - 1):
       LR increases linearly from ``warmup_lr`` to ``base_lr``.
    2. **Cosine decay** (epochs warmup_epochs … total_epochs - 1):
       LR decays from ``base_lr`` down to ``min_lr`` following a half-cosine.

    ``scheduler.step()`` should be called **once per epoch**, not per batch.

    Args:
        optimizer:      Wrapped optimizer.
        warmup_epochs:  Number of linear warm-up epochs.
        total_epochs:   Total number of training epochs.
        base_lr:        Peak learning rate (reached at end of warm-up).
        warmup_lr:      Initial LR at the very start of training.
        min_lr:         Floor LR at the end of cosine decay.
        last_epoch:     The index of the last epoch (for resuming). Default: -1.

    Example::

        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs=20, total_epochs=300,
            base_lr=1e-3, warmup_lr=5e-7, min_lr=5e-6
        )
        for epoch in range(300):
            train(...)
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        warmup_lr: float = 5e-7,
        min_lr: float = 5e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warm-up: warmup_lr → base_lr
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (
                epoch / max(1, self.warmup_epochs)
            )
        else:
            # Cosine decay: base_lr → min_lr
            decay_epochs = self.total_epochs - self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(1, decay_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        return [lr for _ in self.optimizer.param_groups]
