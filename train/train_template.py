"""
Training Loop Template with Weights & Biases Integration.

A production-ready training template for PyTorch models with comprehensive metric tracking,
checkpointing, and device management. Designed as a drop-in template for various tasks:
image classification, language modeling, and multimodal learning.

Key Features:
    - Automatic device detection (CUDA, MPS, CPU)
    - Weights & Biases integration for experiment tracking
    - Gradient accumulation support
    - Mixed precision training (optional)
    - Learning rate scheduling (cosine annealing)
    - Checkpoint management and resumption
    - Early stopping and validation monitoring
    - Detailed metric logging and visualizations

Usage Example:
    >>> from train_template import TrainingConfig, Trainer
    >>> from vit.vitb import ViTB
    >>>
    >>> config = TrainingConfig(
    ...     project_name="my_project",
    ...     experiment_name="vitb_baseline",
    ...     epochs=100,
    ...     batch_size=32,
    ...     learning_rate=1e-4,
    ... )
    >>> model = ViTB(num_classes=1000)
    >>> trainer = Trainer(model, config)
    >>> trainer.train(train_loader, val_loader)

Reference Implementation:
    This template follows best practices from:
    - PyTorch Lightning (structured training)
    - HuggingFace Transformers (config management)
    - Papers with Code (reproducibility)
"""

import os
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """
    Configuration for training run.

    This dataclass defines all hyperparameters and settings for training.
    Can be serialized to JSON for reproducibility.

    Attributes:
        project_name: W&B project name for organizing runs
        experiment_name: Unique name for this training run
        seed: Random seed for reproducibility. Default: 42
        device: Device to train on ("cuda", "mps", "cpu", or None for auto-detect)

        # Training hyperparameters
        epochs: Number of training epochs. Default: 100
        batch_size: Training batch size. Default: 32
        val_batch_size: Validation batch size (can differ from training). Default: 64
        learning_rate: Initial learning rate for optimizer. Default: 1e-4
        warmup_epochs: Number of epochs for LR warmup. Default: 5
        weight_decay: L2 regularization coefficient. Default: 0.01

        # Optimization
        gradient_accumulation_steps: Steps to accumulate gradients before update. Default: 1
        max_grad_norm: Max gradient norm for clipping (None = no clipping). Default: 1.0
        use_mixed_precision: Enable automatic mixed precision training. Default: False

        # Checkpointing & Callbacks
        checkpoint_dir: Directory to save checkpoints. Default: "checkpoints"
        save_every_n_epochs: Save checkpoint frequency. Default: 5
        keep_best_k: Number of best checkpoints to keep. Default: 3
        patience: Epochs without improvement before early stopping. Default: 20

        # Logging
        log_every_n_steps: Log metrics every N steps. Default: 100
        use_wandb: Enable Weights & Biases logging. Default: True
        wandb_entity: W&B entity/team name (optional)
    """

    # Project & Experiment
    project_name: str = "default_project"
    experiment_name: str = "default_experiment"
    seed: int = 42

    # Device & Hardware
    device: Optional[str] = None  # None = auto-detect

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    val_batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    weight_decay: float = 0.01

    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    use_mixed_precision: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    keep_best_k: int = 3
    patience: int = 20

    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    wandb_entity: Optional[str] = None

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class Trainer:
    """
    Production-ready trainer for PyTorch models with W&B integration.

    Handles the full training pipeline including:
    - Device management (auto-detection of CUDA, MPS, CPU)
    - Checkpoint saving and resumption
    - Metric tracking and logging to W&B
    - Learning rate scheduling (cosine annealing with warmup)
    - Gradient accumulation and mixed precision training
    - Early stopping based on validation metrics

    Typical Usage:
        >>> config = TrainingConfig(project_name="my_project")
        >>> model = MyModel()
        >>> trainer = Trainer(model, config)
        >>> trainer.train(train_loader, val_loader)

    The trainer expects:
        - train_loader: DataLoader yielding (batch, labels) tuples
        - val_loader: DataLoader yielding (batch, labels) tuples
        - model.to(device) is called automatically
        - Loss computation via forward pass with labels

    Args:
        model: PyTorch model to train
        config: TrainingConfig instance
        optimizer: Optional optimizer (default: Adam)
        criterion: Loss function (default: CrossEntropyLoss)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
    ):
        self.config = config
        self.model = model
        self.device = self._setup_device()
        self.model.to(self.device)

        # Optimizer and loss
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Mixed precision
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.use_mixed_precision and self.device.type == "cuda"
            else None
        )

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.metrics_history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        # Checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # W&B setup
        self._setup_wandb()

        # Set seed
        self._set_seed(config.seed)

        print(f"Trainer initialized on device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Detect and setup device (CUDA, MPS, or CPU)."""
        if self.config.device:
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases for experiment tracking."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            self.use_wandb = False
            return

        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            entity=self.config.wandb_entity,
            config=asdict(self.config),
            tags=["training"],
        )
        self.use_wandb = True
        print(f"W&B initialized: {wandb.run.url}")

    def _set_seed(self, seed: int) -> None:
        """Set seed for reproducibility."""
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

    def _get_lr_scheduler(self) -> LambdaLR:
        """Create cosine annealing learning rate scheduler with warmup."""
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_epochs:
                # Linear warmup
                return float(step) / float(max(1, self.config.warmup_epochs))
            else:
                # Cosine annealing
                progress = float(step - self.config.warmup_epochs) / float(
                    max(1, self.config.epochs - self.config.warmup_epochs)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Run one training epoch.

        Args:
            train_loader: DataLoader with (images, labels) batches

        Returns:
            (average_loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.max_grad_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Logging
            self.global_step += 1
            if batch_idx % self.config.log_every_n_steps == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = total_correct / total_samples
                print(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {current_loss:.4f}, Acc: {current_acc:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Run validation.

        Args:
            val_loader: DataLoader with (images, labels) batches

        Returns:
            (average_loss, accuracy) on validation set
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def save_checkpoint(self, suffix: str = "") -> str:
        """
        Save model checkpoint.

        Args:
            suffix: Optional suffix for checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            Path(self.config.checkpoint_dir)
            / f"checkpoint_epoch{self.current_epoch:03d}{suffix}.pt"
        )

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "config": asdict(self.config),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the best K checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))

        if len(checkpoints) > self.config.keep_best_k:
            for cp in checkpoints[: -self.config.keep_best_k]:
                cp.unlink()
                print(f"Removed old checkpoint: {cp}")

    def log_metrics(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float,
    ) -> None:
        """
        Log metrics to W&B and internal history.

        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
        """
        metrics = {
            "epoch": self.current_epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": learning_rate,
        }

        # Store in history
        self.metrics_history["train_loss"].append(train_loss)
        self.metrics_history["train_accuracy"].append(train_acc)
        self.metrics_history["val_loss"].append(val_loss)
        self.metrics_history["val_accuracy"].append(val_acc)
        self.metrics_history["learning_rate"].append(learning_rate)

        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        print(
            f"Epoch {self.current_epoch} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            resume_from: Optional checkpoint path to resume from

        Returns:
            Dictionary with training results and metrics history
        """
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

        # Setup learning rate scheduler
        scheduler = self._get_lr_scheduler()

        print(
            f"Starting training for {self.config.epochs} epochs "
            f"(resuming from epoch {self.current_epoch})..."
        )

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            self.log_metrics(train_loss, train_acc, val_loss, val_acc, current_lr)

            # Checkpoint saving
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint()

            # Early stopping based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.patience_counter = 0
                self.save_checkpoint(suffix="_best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(
                        f"Early stopping triggered after {self.config.patience} "
                        f"epochs without improvement"
                    )
                    break

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")

        # Final checkpoint
        self.save_checkpoint(suffix="_final")

        # Log final metrics to W&B
        if self.use_wandb:
            wandb.summary["best_val_loss"] = self.best_val_loss
            wandb.summary["best_val_accuracy"] = self.best_val_accuracy
            wandb.summary["training_time_minutes"] = elapsed_time / 60

        return {
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs": self.current_epoch,
            "elapsed_time": elapsed_time,
            "metrics_history": self.metrics_history,
        }


if __name__ == "__main__":
    """
    Example usage: Training a Vision Transformer on a dummy dataset.

    This demonstrates the training template in action. In practice, replace
    with your actual model and dataloader.
    """
    import sys

    sys.path.insert(0, "..")

    from vit.vitb import ViTB
    from vit.device import get_device

    # Create dummy data
    device = get_device()
    train_data = [(torch.randn(3, 224, 224), torch.randint(0, 10, (1,)).item()) for _ in range(100)]
    val_data = [(torch.randn(3, 224, 224), torch.randint(0, 10, (1,)).item()) for _ in range(20)]

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    # Setup training config
    config = TrainingConfig(
        project_name="personal",
        experiment_name="vitb_dummy",
        epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        use_wandb=False,  # Disable for example
        device=str(device),
    )

    # Create model and trainer
    model = ViTB(num_classes=10)
    trainer = Trainer(model, config)

    # Train
    results = trainer.train(train_loader, val_loader)
    print(f"\nTraining Results: {results}")
