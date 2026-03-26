"""Training entry point for image classification with ResNet18."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.optim import Adam

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from src.config.config import IMAGE_MODELS_DIR
    from src.data.image_dataset import get_dataloaders
    from src.models.image_model import build_image_model
except ModuleNotFoundError:
    # Allow direct execution: python src/training/train_image.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import IMAGE_MODELS_DIR
    from src.data.image_dataset import get_dataloaders
    from src.models.image_model import build_image_model


@dataclass
class TrainConfig:
    """Configuration values for image training."""

    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    pretrained: bool = True
    freeze_backbone: bool = False
    use_progress_bar: bool = True
    early_stopping_patience: int | None = 3
    checkpoint_name: str = "resnet_model.pth"
    seed: int = 42


def get_device() -> torch.device:
    """Select CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _iter_batches(loader: Iterable, use_progress_bar: bool, desc: str) -> Iterable:
    """Wrap dataloader with tqdm when enabled and available."""
    if use_progress_bar and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False)
    return loader


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_progress_bar: bool = True,
) -> float:
    """Run one full training epoch and return mean training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    batch_iterator = _iter_batches(dataloader, use_progress_bar, desc="Train")
    for images, labels in batch_iterator:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def validate_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    use_progress_bar: bool = True,
) -> Tuple[float, float]:
    """Run validation and return mean loss and accuracy percentage."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_iterator = _iter_batches(dataloader, use_progress_bar, desc="Validate")
    with torch.no_grad():
        for images, labels in batch_iterator:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()

    val_loss = running_loss / max(total, 1)
    accuracy = (correct / max(total, 1)) * 100.0
    return val_loss, accuracy


def train_image_classifier(config: TrainConfig) -> Path:
    """Train image classifier and save the best checkpoint path."""
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    num_classes = len(class_to_idx)
    print(f"Detected classes: {num_classes}")

    model = build_image_model(
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    IMAGE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = IMAGE_MODELS_DIR / config.checkpoint_name

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_progress_bar=config.use_progress_bar,
        )
        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_progress_bar=config.use_progress_bar,
        )

        print(
            f"Epoch [{epoch}/{config.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Accuracy: {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "class_to_idx": class_to_idx,
                },
                checkpoint_path,
            )
            print(f"Saved best model to: {checkpoint_path}")
        else:
            patience_counter += 1

        if (
            config.early_stopping_patience is not None
            and patience_counter >= config.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no validation improvement for {config.early_stopping_patience} epochs."
            )
            break

    return checkpoint_path


def run_image_training() -> None:
    """Default training entry point used by scripts and CLI execution."""
    config = TrainConfig()
    saved_path = train_image_classifier(config)
    print(f"Training complete. Best checkpoint: {saved_path}")


if __name__ == "__main__":
    run_image_training()
