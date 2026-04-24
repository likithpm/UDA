"""Multimodal inference pipeline: YOLO video detection + audio classification."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import librosa
import numpy as np
import torch
from ultralytics import YOLO
import yaml

try:
    VideoFileClip = importlib.import_module("moviepy").VideoFileClip
except Exception:
    VideoFileClip = None

try:
    from src.config.config import AUDIO_MODELS_DIR, AQUATIC_DATA_DIR, AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH
    from src.models.audio_model import build_audio_model
except ModuleNotFoundError:
    # Allow direct execution: python src/inference/pipeline.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import AUDIO_MODELS_DIR, AQUATIC_DATA_DIR, AUTO_LABEL_DATA_YAML, YOLO_WEIGHTS_PATH
    from src.models.audio_model import build_audio_model

SAMPLE_RATE = 22050
N_MELS = 128
AUDIO_DURATION_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION_SECONDS
CHUNK_SECONDS = 2
AUDIO_WEIGHTS_PATH = AUDIO_MODELS_DIR / "audio_model.pth"
AudioPrediction = Tuple[float, float, str, float]
AudioTopKPrediction = Tuple[float, float, List[Tuple[str, float]]]
YOLO_FALLBACK_WEIGHTS = "yolov8n.pt"
CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_last_updated(model_path: Path) -> str:
    """Return a readable timestamp for a local model file."""
    try:
        return datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return "unknown"


def _extract_ordered_names_from_yaml(data_yaml: dict) -> list[str]:
    """Return ordered class names from YAML names field."""
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
            raise ValueError("Invalid dict format for names in data.yaml.") from err
    raise ValueError("'names' in data.yaml must be a list or dict.")


def _discover_expected_classes(root_dir: Path) -> list[str]:
    """Discover valid class names from aquatic dataset folders."""
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Aquatic data folder not found: {root_dir}")

    discovered: list[str] = []
    malformed: list[str] = []
    seen_lower: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        class_name = folder.name.strip()
        if not class_name or not CLASS_NAME_PATTERN.fullmatch(class_name):
            malformed.append(folder.name)
            continue

        normalized = class_name.lower()
        existing = seen_lower.get(normalized)
        if existing is not None and existing != class_name:
            duplicates.setdefault(normalized, [existing]).append(class_name)
            continue

        seen_lower[normalized] = class_name
        discovered.append(class_name)

    if malformed:
        raise ValueError(f"Malformed class folder names found: {sorted(set(malformed))}")
    if duplicates:
        duplicate_examples = [sorted(set(names)) for names in duplicates.values()]
        raise ValueError(f"Duplicate class folder names found: {duplicate_examples}")

    sorted_classes = sorted(discovered, key=str.lower)
    if not sorted_classes:
        raise ValueError("No class folders found in aquatic dataset directory.")
    return sorted_classes


def _expected_class_names_from_yaml() -> list[str]:
    """Load and validate class names from dataset_auto_labels/data.yaml."""
    if not AUTO_LABEL_DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found for consistency check: {AUTO_LABEL_DATA_YAML}")

    try:
        data_yaml = yaml.safe_load(AUTO_LABEL_DATA_YAML.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"Invalid YAML format in {AUTO_LABEL_DATA_YAML}: {err}") from err

    if not isinstance(data_yaml, dict):
        raise ValueError(f"Expected mapping in {AUTO_LABEL_DATA_YAML}, found: {type(data_yaml).__name__}")

    yaml_names = _extract_ordered_names_from_yaml(data_yaml)
    yaml_nc = data_yaml.get("nc")
    if not isinstance(yaml_nc, int):
        raise ValueError("'nc' in data.yaml must be an integer.")
    if yaml_nc != len(yaml_names):
        raise ValueError(
            "Class count mismatch in data.yaml: "
            f"nc={yaml_nc}, names_count={len(yaml_names)}"
        )

    folder_names = _discover_expected_classes(AQUATIC_DATA_DIR)
    if yaml_names != folder_names:
        raise ValueError(
            "Class mismatch between aquatic folders and data.yaml. "
            f"folders={folder_names}, data.yaml={yaml_names}"
        )

    return yaml_names


def _model_class_names(yolo_model: YOLO) -> list[str]:
    """Extract ordered class names from loaded YOLO model metadata."""
    names = yolo_model.names
    if isinstance(names, list):
        return [str(name).strip() for name in names]
    if isinstance(names, dict):
        return [
            str(name).strip()
            for _, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    raise ValueError(f"Unsupported YOLO names type: {type(names).__name__}")


def _validate_inference_class_consistency(
    yolo_model: YOLO,
    class_names: list[str],
    strict: bool,
) -> None:
    """Validate class identity/order across folders, data.yaml, and loaded YOLO model."""
    model_names = _model_class_names(yolo_model)

    print(f"Total classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    if len(model_names) != len(class_names):
        message = (
            "Class count mismatch between loaded YOLO model and data.yaml. "
            f"model={len(model_names)}, data.yaml={len(class_names)}"
        )
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")

    if model_names != class_names:
        message = (
            "Class mismatch between loaded YOLO model and data.yaml. "
            f"model={model_names}, expected={class_names}"
        )
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")


def _attach_class_names_to_model(yolo_model: YOLO, class_names: list[str]) -> None:
    """Attach data.yaml class names to model for consistent downstream label mapping."""
    setattr(yolo_model, "class_names_from_yaml", class_names)


def _class_names_for_inference(yolo_model: YOLO) -> list[str]:
    """Get class names for inference labels (always sourced from data.yaml)."""
    class_names = getattr(yolo_model, "class_names_from_yaml", None)
    if isinstance(class_names, list) and class_names:
        return class_names

    # Safety fallback for externally loaded model objects not created via load_models.
    class_names = _expected_class_names_from_yaml()
    _attach_class_names_to_model(yolo_model, class_names)
    return class_names


def _load_yolo_for_inference() -> YOLO:
    """Load trained YOLO weights, with pretrained fallback when missing."""
    class_names = _expected_class_names_from_yaml()

    if YOLO_WEIGHTS_PATH.exists():
        print("Loading existing model...")
        print(f"YOLO model path: {YOLO_WEIGHTS_PATH}")
        print(f"YOLO model last updated: {_model_last_updated(YOLO_WEIGHTS_PATH)}")
        yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
        _validate_inference_class_consistency(yolo_model, class_names=class_names, strict=True)
        _attach_class_names_to_model(yolo_model, class_names)
        return yolo_model

    print(
        "Warning: trained YOLO weights not found at "
        f"{YOLO_WEIGHTS_PATH}. Falling back to pretrained {YOLO_FALLBACK_WEIGHTS}."
    )
    yolo_model = YOLO(YOLO_FALLBACK_WEIGHTS)
    try:
        _validate_inference_class_consistency(yolo_model, class_names=class_names, strict=False)
    except Exception as err:
        print(f"Warning: skipped strict class consistency checks in fallback mode. Details: {err}")
    _attach_class_names_to_model(yolo_model, class_names)
    return yolo_model


def _pad_or_trim(audio: np.ndarray, target_length: int = TARGET_NUM_SAMPLES) -> np.ndarray:
    """Pad short waveform with zeros or trim long waveform to fixed length."""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode="constant")
    if len(audio) > target_length:
        return audio[:target_length]
    return audio


def _normalize(spec: np.ndarray) -> np.ndarray:
    """Apply per-sample z-score normalization."""
    mean = float(np.mean(spec))
    std = float(np.std(spec))
    if std < 1e-8:
        return spec - mean
    return (spec - mean) / std


def _extract_audio_waveform(video_path: Path) -> np.ndarray:
    """Extract mono waveform from video using moviepy, with librosa fallback."""
    if VideoFileClip is not None:
        clip = VideoFileClip(str(video_path))
        try:
            if clip.audio is None:
                raise RuntimeError("No audio track found in the provided video.")

            audio = clip.audio.to_soundarray(fps=SAMPLE_RATE)
            if audio.ndim == 2:
                waveform = np.mean(audio, axis=1)
            else:
                waveform = np.asarray(audio).reshape(-1)

            return waveform.astype(np.float32)
        finally:
            clip.close()

    # Fallback: librosa may load audio stream directly for some video formats.
    waveform, _ = librosa.load(str(video_path), sr=SAMPLE_RATE)
    return waveform


def _waveform_to_mel_tensor(waveform: np.ndarray) -> torch.Tensor:
    """Convert waveform to normalized mel spectrogram tensor: (1, 1, H, W)."""
    waveform = _pad_or_trim(waveform)
    mel = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = _normalize(mel_db)
    return torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)


def _load_audio_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, int]]:
    """Load audio classifier checkpoint and return model with class mapping."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Audio model checkpoint not found: {checkpoint_path}. "
            "Train the audio model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
    else:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing 'class_to_idx'. "
            "Re-train audio model with the provided training script."
        )

    num_classes = len(class_to_idx)
    model = build_audio_model(num_classes=num_classes).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    model.eval()
    return model, class_to_idx


def load_models() -> Tuple[YOLO, torch.nn.Module, Dict[str, int], torch.device, str]:
    """Load YOLO and audio models along with class mapping and device info."""
    device = _get_device()
    yolo_device = "0" if device.type == "cuda" else "cpu"

    yolo_model = _load_yolo_for_inference()
    audio_model, class_to_idx = _load_audio_checkpoint(AUDIO_WEIGHTS_PATH, device=device)

    return yolo_model, audio_model, class_to_idx, device, yolo_device


def process_audio(video_path: Path, audio_model: torch.nn.Module, class_to_idx: Dict[str, int], device: torch.device) -> Tuple[str, float]:
    """Backward-compatible single-label audio prediction helper."""
    predictions = process_audio_chunks(video_path, audio_model, class_to_idx, device)
    if not predictions:
        raise RuntimeError("No audio predictions generated from the input video.")

    # Use the highest-confidence chunk as the global summary.
    _, _, label, confidence = max(predictions, key=lambda item: item[3])
    return label, confidence


def process_audio_chunks(
    video_path: Path,
    audio_model: torch.nn.Module,
    class_to_idx: Dict[str, int],
    device: torch.device,
) -> List[AudioPrediction]:
    """Predict top-1 audio labels for consecutive 2-second chunks of video audio."""
    topk_predictions = process_audio_chunks_topk(
        video_path=video_path,
        audio_model=audio_model,
        class_to_idx=class_to_idx,
        device=device,
        top_k=1,
    )

    audio_predictions: List[AudioPrediction] = []
    for start_time, end_time, topk in topk_predictions:
        label, confidence = topk[0]
        audio_predictions.append((start_time, end_time, label, confidence))

    return audio_predictions


def process_audio_chunks_topk(
    video_path: Path,
    audio_model: torch.nn.Module,
    class_to_idx: Dict[str, int],
    device: torch.device,
    top_k: int = 3,
) -> List[AudioTopKPrediction]:
    """Predict top-k audio labels for each consecutive 2-second chunk."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    waveform = _extract_audio_waveform(video_path)
    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    audio_predictions: List[AudioTopKPrediction] = []
    max_k = min(top_k, len(idx_to_class))

    total_samples = len(waveform)
    if total_samples == 0:
        raise RuntimeError("Extracted audio waveform is empty.")

    for start_sample in range(0, total_samples, chunk_size):
        end_sample = min(start_sample + chunk_size, total_samples)
        chunk = waveform[start_sample:end_sample]
        if chunk.size == 0:
            continue

        mel_tensor = _waveform_to_mel_tensor(chunk).to(device)

        with torch.no_grad():
            logits = audio_model(mel_tensor)
            probabilities = torch.softmax(logits, dim=1)
            top_confidences, top_indices = torch.topk(probabilities, k=max_k, dim=1)

        topk_predictions = [
            (idx_to_class[int(class_idx.item())], float(class_conf.item()))
            for class_conf, class_idx in zip(top_confidences[0], top_indices[0])
        ]
        start_time = start_sample / SAMPLE_RATE
        end_time = end_sample / SAMPLE_RATE
        audio_predictions.append((start_time, end_time, topk_predictions))

    if not audio_predictions:
        raise RuntimeError("No audio chunks were processed for prediction.")

    return audio_predictions


def _prediction_for_timestamp(
    timestamp_seconds: float,
    audio_predictions: List[AudioPrediction],
) -> AudioPrediction:
    """Return the matching audio chunk prediction for the current timestamp."""
    for prediction in audio_predictions:
        start_time, end_time, _, _ = prediction
        if start_time <= timestamp_seconds < end_time:
            return prediction

    # Fallback to last prediction when timestamp exceeds final chunk due to decode timing.
    return audio_predictions[-1]



def _draw_yolo_detections(frame: np.ndarray, result, class_names: list[str]) -> None:
    """Draw YOLO bounding boxes and labels onto a frame."""
    boxes = result.boxes
    if boxes is None:
        return

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        if cls_id < 0 or cls_id >= len(class_names):
            print(
                "Warning: class_id out of range during annotation; skipping detection. "
                f"class_id={cls_id}, allowed=[0, {len(class_names) - 1}]"
            )
            continue

        conf = float(box.conf[0].item())
        label = class_names[cls_id]

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (p1[0], max(0, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def annotate_frame_with_yolo(frame: np.ndarray, yolo_model: YOLO, yolo_device: str) -> np.ndarray:
    """Run YOLO inference and return the frame with rendered detections."""
    annotated = frame.copy()
    class_names = _class_names_for_inference(yolo_model)
    results = yolo_model.predict(source=annotated, verbose=False, device=yolo_device)
    if results:
        _draw_yolo_detections(annotated, results[0], class_names)
    return annotated


def annotate_and_collect_objects(
    frame: np.ndarray,
    yolo_model: YOLO,
    yolo_device: str,
    conf_threshold: float = 0.3,
    inference_size: int | None = None,
    copy_frame: bool = True,
) -> Tuple[np.ndarray, Set[str], List[Tuple[str, float]]]:
    """Run YOLO inference and return frame, object names, and (label, confidence) list."""
    annotated = frame.copy() if copy_frame else frame
    class_names = _class_names_for_inference(yolo_model)
    detected_objects: Set[str] = set()
    detected_with_confidence: List[Tuple[str, float]] = []

    inference_frame = annotated
    scale_x = 1.0
    scale_y = 1.0
    if inference_size and inference_size > 0:
        source_height, source_width = annotated.shape[:2]
        inference_frame = cv2.resize(
            annotated,
            (inference_size, inference_size),
            interpolation=cv2.INTER_AREA,
        )
        scale_x = source_width / float(inference_size)
        scale_y = source_height / float(inference_size)

    results = yolo_model.predict(
        source=inference_frame,
        verbose=False,
        device=yolo_device,
        conf=conf_threshold,
    )

    if results:
        result = results[0]

        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id < 0 or cls_id >= len(class_names):
                    print(
                        "Warning: class_id out of range during object collection; skipping detection. "
                        f"class_id={cls_id}, allowed=[0, {len(class_names) - 1}]"
                    )
                    continue

                conf = float(box.conf[0].item())
                label = class_names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if inference_frame is not annotated:
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y

                frame_h, frame_w = annotated.shape[:2]
                p1 = (max(0, min(int(x1), frame_w - 1)), max(0, min(int(y1), frame_h - 1)))
                p2 = (max(0, min(int(x2), frame_w - 1)), max(0, min(int(y2), frame_h - 1)))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (p1[0], max(0, p1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                detected_objects.add(label)
                detected_with_confidence.append((label, conf))

    return annotated, detected_objects, detected_with_confidence


def run_video(
    video_path: Path,
    yolo_model: YOLO,
    yolo_device: str,
    audio_predictions: List[AudioPrediction],
) -> None:
    """Run YOLO detection and show synchronized dynamic audio labels per frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video stream ended.")
                break

            frame = annotate_frame_with_yolo(frame, yolo_model=yolo_model, yolo_device=yolo_device)

            timestamp_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            start_t, end_t, label, confidence = _prediction_for_timestamp(
                timestamp_seconds=timestamp_seconds,
                audio_predictions=audio_predictions,
            )
            cv2.putText(
                frame,
                f"Audio: {label} ({confidence:.2f}) [{start_t:.1f}-{end_t:.1f}s]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Multimodal AI Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exit requested by user.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_pipeline(video_path: str) -> None:
    """Execute audio prediction first, then run frame-level YOLO detection."""
    input_video = Path(video_path)
    if not input_video.exists() or not input_video.is_file():
        raise FileNotFoundError(f"Invalid video path: {input_video}")

    yolo_model, audio_model, class_to_idx, device, yolo_device = load_models()

    audio_predictions = process_audio_chunks(
        video_path=input_video,
        audio_model=audio_model,
        class_to_idx=class_to_idx,
        device=device,
    )
    first_start, first_end, first_label, first_confidence = audio_predictions[0]
    print(
        "Audio prediction (initial chunk): "
        f"{first_label} (confidence: {first_confidence:.2f}) "
        f"[{first_start:.1f}-{first_end:.1f}s]"
    )

    run_video(
        video_path=input_video,
        yolo_model=yolo_model,
        yolo_device=yolo_device,
        audio_predictions=audio_predictions,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multimodal pipeline execution."""
    parser = argparse.ArgumentParser(description="Run multimodal video+audio AI inference.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.video)
