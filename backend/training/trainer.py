"""Trainer class for Swin Transformer.

Wires together the model, optimizer, LR scheduler, loss function,
mixed-precision training (AMP), gradient clipping, MixUp/CutMix
augmentation, checkpointing, and optional Weights & Biases logging.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast  # type: ignore[attr-defined]

from .losses import (
    LabelSmoothingCrossEntropy,
    cutmix_data,
    mixup_criterion,
    mixup_data,
)
from .scheduler import CosineWarmupScheduler


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TrainerConfig:
    """Hyper-parameters that control the Trainer.

    All fields have sane defaults matching the Swin-T CIFAR-10 recipe.
    """

    # Optimisation
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.05

    # LR schedule
    warmup_epochs: int = 20
    warmup_lr: float = 5e-7
    min_lr: float = 5e-6

    # Gradient clipping
    grad_clip: float = 5.0

    # Mixed precision (safe no-op on CPU when enabled=False internally)
    amp: bool = True

    # Regularisation
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5  # probability of applying MixUp OR CutMix per batch

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Logging
    use_wandb: bool = False


# ── Trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    """Full training loop for Swin Transformer image classification.

    Args:
        model:   An ``nn.Module`` (any SwinTransformer variant).
        config:  ``TrainerConfig`` controlling all hyper-parameters.
        device:  ``"cpu"`` or ``"cuda"`` (or any valid torch device string).

    Example::

        config = TrainerConfig(epochs=2, use_wandb=False)
        trainer = Trainer(model, config, device="cpu")
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler (stepped once per epoch)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
            base_lr=config.lr,
            warmup_lr=config.warmup_lr,
            min_lr=config.min_lr,
        )

        # Loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)

        # AMP scaler — enabled=False is a silent no-op on CPU
        self.scaler = GradScaler(enabled=config.amp and device != "cpu")

        # Best validation accuracy seen so far
        self.best_acc: float = 0.0

        # Ensure checkpoint directory exists
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── single epoch ──────────────────────────────────────────────────────────

    def train_one_epoch(self, loader) -> float:
        """Run one full pass over ``loader`` in training mode.

        Args:
            loader: A ``DataLoader`` yielding ``(images, labels)`` batches.

        Returns:
            Average training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # ── Augmentation decision ─────────────────────────────────────
            use_mix = torch.rand(1).item() < self.config.mix_prob
            if use_mix:
                use_cutmix = torch.rand(1).item() < 0.5
                if use_cutmix:
                    images, y_a, y_b, lam = cutmix_data(
                        images, labels, alpha=self.config.cutmix_alpha
                    )
                else:
                    images, y_a, y_b, lam = mixup_data(
                        images, labels, alpha=self.config.mixup_alpha
                    )
            # ─────────────────────────────────────────────────────────────

            with autocast(enabled=self.config.amp and self.device != "cpu"):
                logits = self.model(images)
                if use_mix:
                    loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first so the norm is in true scale)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, loader) -> tuple[float, float]:
        """Evaluate the model on ``loader`` without augmentation.

        Args:
            loader: A ``DataLoader`` yielding ``(images, labels)`` batches.

        Returns:
            ``(val_loss, top1_accuracy)`` where accuracy is in percent [0, 100].
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / max(1, len(loader))
        top1_acc = 100.0 * correct / max(1, total)
        return val_loss, top1_acc

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False) -> None:
        """Persist model and optimizer state to disk.

        Always writes to ``<checkpoint_dir>/last.pth``.
        When ``is_best`` is True also writes to ``<checkpoint_dir>/best.pth``.

        Args:
            path:    Destination path for *this* epoch's checkpoint.
            epoch:   Current epoch index (0-based), stored in the payload.
            is_best: Whether this epoch achieved the best val accuracy so far.
        """
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
        }

        last_path = os.path.join(self.config.checkpoint_dir, "last.pth")
        torch.save(payload, last_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pth")
            torch.save(payload, best_path)

    # ── main loop ─────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        on_epoch_end: Optional[Callable[[int, dict], None]] = None,
    ) -> None:
        """Run the full training loop for ``config.epochs`` epochs.

        After each epoch the scheduler is stepped, a checkpoint is saved,
        and ``on_epoch_end(epoch, metrics)`` is called if provided.

        Args:
            train_loader:  DataLoader for training data.
            val_loader:    DataLoader for validation data.
            on_epoch_end:  Optional callback with signature
                           ``(epoch: int, metrics: dict) -> None``.
                           The metrics dict has keys:
                           ``train_loss``, ``val_loss``, ``val_acc``, ``lr``.
        """
        # Lazy W&B import — never required at module level
        _wandb = None
        if self.config.use_wandb:
            try:
                import wandb as _wandb  # type: ignore[import]
            except ImportError:
                print("[Trainer] wandb not installed — skipping W&B logging.")
                _wandb = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Advance LR schedule
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            # Always write last.pth; write best.pth when accuracy improves
            ckpt_path = os.path.join(
                self.config.checkpoint_dir, f"epoch_{epoch:04d}.pth"
            )
            self.save_checkpoint(ckpt_path, epoch, is_best=is_best)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }

            # Console summary
            print(
                f"Epoch [{epoch + 1:>4d}/{self.config.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2f}%  "
                f"lr={current_lr:.2e}"
                + ("  ★ best" if is_best else "")
            )

            # Optional W&B logging
            if _wandb is not None:
                _wandb.log({"epoch": epoch, **metrics})

            # Optional caller callback (used by the API layer for live status)
            if on_epoch_end is not None:
                on_epoch_end(epoch, metrics)
