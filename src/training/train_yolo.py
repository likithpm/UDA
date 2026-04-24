"""Train a YOLOv8 detector using ultralytics and project config paths."""

from __future__ import annotations

from datetime import datetime
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from ultralytics import YOLO

try:
    from src.config.config import (
        AUTO_LABEL_DATA_YAML,
        AQUATIC_DATA_DIR,
        BASE_DIR,
        DETECTION_MODELS_DIR,
        YOLO_WEIGHTS_PATH,
    )
except ModuleNotFoundError:
    # Allow direct execution: python src/training/train_yolo.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import (
        AUTO_LABEL_DATA_YAML,
        AQUATIC_DATA_DIR,
        BASE_DIR,
        DETECTION_MODELS_DIR,
        YOLO_WEIGHTS_PATH,
    )

CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _resolve_device() -> str:
    """Return device string for ultralytics training."""
    return "0" if torch.cuda.is_available() else "cpu"


def _model_last_updated(model_path: Path) -> str:
    """Return a readable timestamp for model artifact metadata."""
    try:
        return datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return "unknown"


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


def _extract_ordered_names(data_yaml: Dict[str, Any]) -> list[str]:
    """Extract ordered class names from data.yaml 'names' field."""
    names_field = data_yaml.get("names", [])
    if isinstance(names_field, list):
        return [str(name).strip() for name in names_field]
    if isinstance(names_field, dict):
        try:
            return [
                str(name).strip()
                for _, name in sorted(
                    ((int(idx), value) for idx, value in names_field.items()),
                    key=lambda item: item[0],
                )
            ]
        except (TypeError, ValueError) as err:
            raise ValueError("Invalid dict format for 'names' in data.yaml.") from err
    raise ValueError("'names' in data.yaml must be a list or dict.")


def _discover_expected_classes(root_dir: Path) -> list[str]:
    """Discover valid class folders with deterministic ordering."""
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Aquatic data folder not found: {root_dir}")

    discovered: list[str] = []
    malformed: list[str] = []
    seen_normalized: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        class_name = folder.name.strip()
        if not class_name or not CLASS_NAME_PATTERN.fullmatch(class_name):
            malformed.append(folder.name)
            continue

        normalized = class_name.lower()
        existing = seen_normalized.get(normalized)
        if existing is not None and existing != class_name:
            duplicates.setdefault(normalized, [existing]).append(class_name)
            continue

        seen_normalized[normalized] = class_name
        discovered.append(class_name)

    if malformed:
        raise ValueError(
            "Malformed class folder names found in aquatic dataset. "
            f"Invalid: {sorted(set(malformed))}"
        )
    if duplicates:
        duplicate_examples = [sorted(set(names)) for names in duplicates.values()]
        raise ValueError(
            "Duplicate class folder names detected (case-insensitive): "
            f"{duplicate_examples}"
        )

    sorted_classes = sorted(discovered, key=str.lower)
    if not sorted_classes:
        raise ValueError("No class folders found under aquatic dataset directory.")
    return sorted_classes


def _validate_and_sanitize_labels(
    labels_dir: Path,
    class_names: list[str],
) -> dict[str, int]:
    """Validate label class ids and keep only lines in [0, nc-1]."""
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    num_classes = len(class_names)
    labels_per_class = {class_name: 0 for class_name in class_names}
    invalid_line_count = 0

    label_files = sorted(labels_dir.rglob("*.txt"))
    for label_file in label_files:
        original_text = label_file.read_text(encoding="utf-8")
        cleaned_lines: list[str] = []

        for line_no, raw_line in enumerate(original_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                invalid_line_count += 1
                print(
                    "Error: malformed label line; skipping. "
                    f"file={label_file}, line={line_no}, content='{line}'"
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                invalid_line_count += 1
                print(
                    "Error: non-integer class id; skipping. "
                    f"file={label_file}, line={line_no}, class_id='{parts[0]}'"
                )
                continue

            if class_id < 0 or class_id >= num_classes:
                invalid_line_count += 1
                print(
                    "Error: class_id out of range; skipping. "
                    f"file={label_file}, line={line_no}, class_id={class_id}, "
                    f"allowed=[0, {num_classes - 1}]"
                )
                continue

            cleaned_lines.append(line)
            labels_per_class[class_names[class_id]] += 1

        normalized_text = "\n".join(cleaned_lines)
        if normalized_text:
            normalized_text += "\n"

        if normalized_text != original_text:
            label_file.write_text(normalized_text, encoding="utf-8")

    if invalid_line_count:
        print(f"Skipped {invalid_line_count} invalid label line(s) during validation.")

    return labels_per_class


def _resolve_dataset_root(data_yaml: Dict[str, Any], data_yaml_path: Path) -> Path:
    """Resolve dataset root using YAML 'path' with project-relative fallback."""
    path_value = data_yaml.get("path")
    if not path_value:
        return data_yaml_path.parent

    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate

    # Project config stores path as project-relative.
    return BASE_DIR / candidate


def _resolve_split_dir(dataset_root: Path, split_value: Any) -> Path:
    """Resolve train/val folder path from YAML entry."""
    if not split_value:
        raise ValueError("Missing split path in data.yaml.")
    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def _clear_yolo_cache_files(dataset_root: Path) -> int:
    """Delete stale YOLO .cache files to avoid inconsistent cached metadata."""
    deleted_count = 0
    for cache_file in dataset_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            deleted_count += 1
        except OSError as err:
            print(f"Warning: failed to delete cache file {cache_file}: {err}")
    return deleted_count


def validate_yolo_dataset() -> None:
    """Run dry-run checks for YOLO dataset config before training."""
    if not AUTO_LABEL_DATA_YAML.exists():
        raise FileNotFoundError(
            f"YOLO data config not found: {AUTO_LABEL_DATA_YAML}. "
            "Run auto annotation first to generate dataset_auto_labels/data.yaml."
        )

    data_yaml = _load_data_yaml(AUTO_LABEL_DATA_YAML)
    dataset_root = _resolve_dataset_root(data_yaml, AUTO_LABEL_DATA_YAML)
    train_images_dir = _resolve_split_dir(dataset_root, data_yaml.get("train"))
    val_images_dir = _resolve_split_dir(dataset_root, data_yaml.get("val"))

    if not train_images_dir.exists() or not train_images_dir.is_dir():
        raise FileNotFoundError(f"Train images folder not found: {train_images_dir}")
    if not val_images_dir.exists() or not val_images_dir.is_dir():
        raise FileNotFoundError(f"Validation images folder not found: {val_images_dir}")

    yaml_class_names = _extract_ordered_names(data_yaml)
    class_count = len(yaml_class_names)
    expected_classes = _discover_expected_classes(AQUATIC_DATA_DIR)
    yaml_nc = data_yaml.get("nc")

    if class_count <= 0:
        raise ValueError("No classes found in data.yaml. Ensure 'names' contains at least one class.")
    if not isinstance(yaml_nc, int):
        raise ValueError("'nc' in data.yaml must be an integer.")
    if yaml_nc != class_count:
        raise ValueError(
            "Class count mismatch in data.yaml: "
            f"nc={yaml_nc}, names_count={class_count}"
        )
    if yaml_class_names != expected_classes:
        raise ValueError(
            "Class mismatch between aquatic folder names and data.yaml names/order. "
            f"folders={expected_classes}, data.yaml={yaml_class_names}"
        )

    labels_dir = dataset_root / "labels"
    labels_per_class = _validate_and_sanitize_labels(labels_dir, yaml_class_names)

    print("Dataset summary:")
    print(f"- data.yaml: {AUTO_LABEL_DATA_YAML}")
    print(f"- dataset root: {dataset_root}")
    print(f"- train images: {train_images_dir}")
    print(f"- val images: {val_images_dir}")
    print(f"- class count: {class_count}")
    print(f"- class names: {yaml_class_names}")
    print("- labels per class:")
    for class_name in yaml_class_names:
        print(f"  - {class_name}: {labels_per_class[class_name]}")
    print("Dataset validation successful. Ready for training.")


def run_yolo_training(force_train: bool = False) -> Path:
    """Train YOLOv8n with configured dataset and save best model checkpoint."""
    if not force_train and YOLO_WEIGHTS_PATH.exists():
        print("Using existing trained model")
        print("Loading existing model...")
        print(f"Model path: {YOLO_WEIGHTS_PATH}")
        print(f"Model last updated: {_model_last_updated(YOLO_WEIGHTS_PATH)}")
        return YOLO_WEIGHTS_PATH

    print("Training new model...")
    if force_train and YOLO_WEIGHTS_PATH.exists():
        print("force_train=True -> existing model will be replaced after training.")

    validate_yolo_dataset()

    data_yaml = _load_data_yaml(AUTO_LABEL_DATA_YAML)
    dataset_root = _resolve_dataset_root(data_yaml, AUTO_LABEL_DATA_YAML)

    # Clear YOLO cache files before training to prevent stale cached shapes/labels.
    deleted_cache_files = _clear_yolo_cache_files(dataset_root)
    if deleted_cache_files:
        print(f"Cleared {deleted_cache_files} YOLO cache file(s) from: {dataset_root}")

    device = _resolve_device()
    print(f"Using device: {device}")

    # Requirement: start from lightweight pretrained YOLOv8n weights.
    model = YOLO("yolov8n.pt")

    """
    NOTE:
    This training uses auto-generated bounding box labels created via pretrained YOLO.

    This significantly improves localization compared to full-image labels.
    However, label quality depends on the initial auto-detection accuracy.

    Further improvement can be achieved via iterative re-labeling.
    """

    results = model.train(
        data=str(AUTO_LABEL_DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        augment=True,
        multi_scale=False,
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
    print(f"Model last updated: {_model_last_updated(final_weights_path)}")
    return final_weights_path


if __name__ == "__main__":
    force_train = True
    run_yolo_training(force_train=force_train)
