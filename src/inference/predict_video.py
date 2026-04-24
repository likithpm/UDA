"""Real-time video detection using a trained YOLOv8 model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO
import yaml

try:
    from src.config.config import AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH
except ModuleNotFoundError:
    # Allow direct execution: python src/inference/predict_video.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH


def _extract_ordered_names_from_yaml(data_yaml: dict) -> list[str]:
    """Extract ordered class names from names field in data.yaml."""
    names_field = data_yaml.get("names", [])
    if isinstance(names_field, list):
        return [str(name).strip() for name in names_field]
    if isinstance(names_field, dict):
        return [
            str(name).strip()
            for _, name in sorted(
                ((int(idx), value) for idx, value in names_field.items()),
                key=lambda item: item[0],
            )
        ]
    raise ValueError("'names' in data.yaml must be a list or dict.")


def _load_class_names_from_data_yaml() -> list[str]:
    """Load class names from dataset_auto_labels/data.yaml only."""
    if not AUTO_LABEL_DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {AUTO_LABEL_DATA_YAML}")

    try:
        data_yaml = yaml.safe_load(AUTO_LABEL_DATA_YAML.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"Invalid YAML format in {AUTO_LABEL_DATA_YAML}: {err}") from err

    if not isinstance(data_yaml, dict):
        raise ValueError(f"Expected mapping in {AUTO_LABEL_DATA_YAML}, found: {type(data_yaml).__name__}")

    class_names = _extract_ordered_names_from_yaml(data_yaml)
    yaml_nc = data_yaml.get("nc")
    if not isinstance(yaml_nc, int) or yaml_nc != len(class_names):
        raise ValueError(
            "Invalid class metadata in data.yaml: "
            f"nc={yaml_nc}, names_count={len(class_names)}"
        )

    return class_names


def _model_class_count(model: YOLO) -> int:
    """Return class count from YOLO model metadata."""
    names = model.names
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    raise ValueError(f"Unsupported YOLO names type: {type(names).__name__}")


def load_model(weights_path: Path) -> tuple[YOLO, list[str]]:
    """Load YOLO model from configured weights path."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained YOLO weights not found: {weights_path}. "
            "Train YOLO first to generate yolo_weights.pt."
        )

    class_names = _load_class_names_from_data_yaml()
    model = YOLO(str(weights_path))
    model_class_count = _model_class_count(model)

    print(f"Total classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    if model_class_count != len(class_names):
        raise ValueError(
            "Class count mismatch between model and data.yaml. "
            f"model={model_class_count}, data.yaml={len(class_names)}"
        )

    return model, class_names


def open_video_source(video_path: Optional[str] = None) -> cv2.VideoCapture:
    """Open webcam by default, or a video file when provided."""
    source = 0 if not video_path else video_path
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return capture


def _draw_predictions(frame, result, class_names: list[str]) -> None:
    """Draw bounding boxes with class label and confidence on frame."""
    boxes = result.boxes
    if boxes is None:
        return

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        if class_id < 0 or class_id >= len(class_names):
            print(
                "Warning: class_id out of range during video prediction; skipping detection. "
                f"class_id={class_id}, allowed=[0, {len(class_names) - 1}]"
            )
            continue

        conf = float(box.conf[0].item())
        label = class_names[class_id]
        text = f"{label} {conf:.2f}"

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (p1[0], max(0, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def predict_video(video_path: Optional[str] = None) -> None:
    """Run real-time YOLO detection on webcam or video file."""
    model, class_names = load_model(YOLO_WEIGHTS_PATH)
    capture = open_video_source(video_path)

    prev_time = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("No more frames available or failed to read frame.")
                break

            results = model.predict(source=frame, verbose=False)
            if results:
                _draw_predictions(frame, results[0], class_names)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-9)
            prev_time = current_time
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLOv8 Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exit requested by user.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for optional video path."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 real-time video detection.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file. If omitted, webcam is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict_video(video_path=args.video)
