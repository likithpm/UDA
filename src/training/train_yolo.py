"""Train a YOLOv8 detector using ultralytics and project config paths."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from ultralytics import YOLO

try:
    from src.config.config import (
        BASE_DIR,
        DETECTION_MODELS_DIR,
        YOLO_DATA_YAML,
        YOLO_WEIGHTS_PATH,
    )
except ModuleNotFoundError:
    # Allow direct execution: python src/training/train_yolo.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import (
        BASE_DIR,
        DETECTION_MODELS_DIR,
        YOLO_DATA_YAML,
        YOLO_WEIGHTS_PATH,
    )


def _resolve_device() -> str:
    """Return device string for ultralytics training."""
    return "0" if torch.cuda.is_available() else "cpu"


def _copy_best_weights(best_source: Path, destination: Path) -> Path:
    """Copy best trained YOLO weights into project model directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, destination)
    return destination


def _load_data_yaml(data_yaml_path: Path) -> Dict[str, Any]:
    """Load YOLO data YAML into a dictionary."""
    try:
        yaml_data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"Invalid YAML format in {data_yaml_path}: {err}") from err

    if not isinstance(yaml_data, dict):
        raise ValueError(f"Expected mapping in {data_yaml_path}, found: {type(yaml_data).__name__}")
    return yaml_data


def _resolve_dataset_root(data_yaml: Dict[str, Any], data_yaml_path: Path) -> Path:
    """Resolve dataset root using YAML 'path' with project-relative fallback."""
    path_value = data_yaml.get("path")
    if not path_value:
        return data_yaml_path.parent

    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate

    # Our project stores path as project-relative, e.g. "dataset_yolo".
    return BASE_DIR / candidate


def _resolve_split_dir(dataset_root: Path, split_value: Any) -> Path:
    """Resolve train/val folder path from YAML entry."""
    if not split_value:
        raise ValueError("Missing split path in data.yaml.")
    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def validate_yolo_dataset() -> None:
    """Run dry-run checks for YOLO dataset config before training."""
    if not YOLO_DATA_YAML.exists():
        raise FileNotFoundError(
            f"YOLO data config not found: {YOLO_DATA_YAML}. "
            "Run the dataset conversion script first."
        )

    data_yaml = _load_data_yaml(YOLO_DATA_YAML)
    dataset_root = _resolve_dataset_root(data_yaml, YOLO_DATA_YAML)
    train_images_dir = _resolve_split_dir(dataset_root, data_yaml.get("train"))
    val_images_dir = _resolve_split_dir(dataset_root, data_yaml.get("val"))

    if not train_images_dir.exists() or not train_images_dir.is_dir():
        raise FileNotFoundError(f"Train images folder not found: {train_images_dir}")
    if not val_images_dir.exists() or not val_images_dir.is_dir():
        raise FileNotFoundError(f"Validation images folder not found: {val_images_dir}")

    names = data_yaml.get("names", [])
    if isinstance(names, dict):
        class_count = len(names)
    elif isinstance(names, list):
        class_count = len(names)
    else:
        raise ValueError("'names' in data.yaml must be a list or dict.")

    if class_count <= 0:
        raise ValueError("No classes found in data.yaml. Ensure 'names' contains at least one class.")

    print("Dataset summary:")
    print(f"- data.yaml: {YOLO_DATA_YAML}")
    print(f"- dataset root: {dataset_root}")
    print(f"- train images: {train_images_dir}")
    print(f"- val images: {val_images_dir}")
    print(f"- class count: {class_count}")
    print("Dataset validation successful. Ready for training.")


def run_yolo_training() -> None:
    """Train YOLOv8n with configured dataset and save best model checkpoint."""
    validate_yolo_dataset()

    device = _resolve_device()
    print(f"Using device: {device}")

    # Requirement: start from lightweight pretrained YOLOv8n weights.
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        augment=True,
        device=device,
        project=str(DETECTION_MODELS_DIR),
        name="yolo_train",
        exist_ok=True,
        verbose=True,
    )

    print("Training finished.")
    print(f"Training results summary: {results}")

    best_weights_source = Path(model.trainer.best)
    if not best_weights_source.exists():
        raise FileNotFoundError(f"Best weights were not found at: {best_weights_source}")

    final_weights_path = _copy_best_weights(best_weights_source, YOLO_WEIGHTS_PATH)
    print(f"Best model saved to: {final_weights_path}")


if __name__ == "__main__":
    run_yolo_training()
