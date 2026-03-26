"""Real-time video detection using a trained YOLOv8 model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

try:
    from src.config.config import YOLO_WEIGHTS_PATH
except ModuleNotFoundError:
    # Allow direct execution: python src/inference/predict_video.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import YOLO_WEIGHTS_PATH


def load_model(weights_path: Path) -> YOLO:
    """Load YOLO model from configured weights path."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained YOLO weights not found: {weights_path}. "
            "Train YOLO first to generate yolo_weights.pt."
        )
    return YOLO(str(weights_path))


def open_video_source(video_path: Optional[str] = None) -> cv2.VideoCapture:
    """Open webcam by default, or a video file when provided."""
    source = 0 if not video_path else video_path
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return capture


def _draw_predictions(frame, result) -> None:
    """Draw bounding boxes with class label and confidence on frame."""
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names.get(class_id, str(class_id))
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
    model = load_model(YOLO_WEIGHTS_PATH)
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
                _draw_predictions(frame, results[0])

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
