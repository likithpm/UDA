"""Training entry point for audio classification with a CNN model."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.optim import Adam

try:
    from src.config.config import AUDIO_MODELS_DIR
    from src.data.audio_dataset import get_dataloaders
    from src.models.audio_model import build_audio_model
except ModuleNotFoundError:
    # Allow direct execution: python src/training/train_audio.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import AUDIO_MODELS_DIR
    from src.data.audio_dataset import get_dataloaders
    from src.models.audio_model import build_audio_model


@dataclass
class TrainConfig:
    """Configuration for audio model training."""

    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-3
    early_stopping_patience: int | None = 5
    checkpoint_name: str = "audio_model.pth"
    seed: int = 42


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
) -> Tuple[float, float]:
    """Run validation epoch and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    val_loss = running_loss / max(total, 1)
    accuracy = (correct / max(total, 1)) * 100.0
    return val_loss, accuracy


def train_audio_classifier(config: TrainConfig) -> Path:
    """Train audio classifier and save the best model checkpoint."""
    device = get_device()
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    num_classes = len(class_to_idx)
    print(f"[INFO] Number of classes: {num_classes}")

    model = build_audio_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    AUDIO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = AUDIO_MODELS_DIR / config.checkpoint_name

    best_val_loss = float("inf")
    patience_counter = 0

    print("[INFO] Starting training...")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, accuracy = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
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
            print(f"[INFO] Best model updated: {checkpoint_path}")
        else:
            patience_counter += 1

        if (
            config.early_stopping_patience is not None
            and patience_counter >= config.early_stopping_patience
        ):
            print(
                "[INFO] Early stopping triggered: "
                f"no validation improvement for {config.early_stopping_patience} epochs."
            )
            break

    print(f"[INFO] Training complete. Best checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_audio_training() -> None:
    """Default entry point for audio model training."""
    config = TrainConfig()
    train_audio_classifier(config)


if __name__ == "__main__":
    run_audio_training()
