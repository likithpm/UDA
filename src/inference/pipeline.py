"""Multimodal inference pipeline: YOLO video detection + audio classification."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import librosa
import numpy as np
import torch
from ultralytics import YOLO

try:
    VideoFileClip = importlib.import_module("moviepy").VideoFileClip
except Exception:
    VideoFileClip = None

try:
    from src.config.config import AUDIO_MODELS_DIR, YOLO_WEIGHTS_PATH
    from src.models.audio_model import build_audio_model
except ModuleNotFoundError:
    # Allow direct execution: python src/inference/pipeline.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.config.config import AUDIO_MODELS_DIR, YOLO_WEIGHTS_PATH
    from src.models.audio_model import build_audio_model

SAMPLE_RATE = 22050
N_MELS = 128
AUDIO_DURATION_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION_SECONDS
CHUNK_SECONDS = 2
AUDIO_WEIGHTS_PATH = AUDIO_MODELS_DIR / "audio_model.pth"
AudioPrediction = Tuple[float, float, str, float]
AudioTopKPrediction = Tuple[float, float, List[Tuple[str, float]]]


def _get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_WEIGHTS_PATH}. Train YOLO model first."
        )

    device = _get_device()
    yolo_device = "0" if device.type == "cuda" else "cpu"

    yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
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



def _draw_yolo_detections(frame: np.ndarray, result) -> None:
    """Draw YOLO bounding boxes and labels onto a frame."""
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names.get(cls_id, str(cls_id))

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
    results = yolo_model.predict(source=annotated, verbose=False, device=yolo_device)
    if results:
        _draw_yolo_detections(annotated, results[0])
    return annotated


def annotate_and_collect_objects(
    frame: np.ndarray,
    yolo_model: YOLO,
    yolo_device: str,
    conf_threshold: float = 0.25,
) -> Tuple[np.ndarray, Set[str]]:
    """Run YOLO inference, return annotated frame and detected class-name set."""
    annotated = frame.copy()
    detected_objects: Set[str] = set()
    results = yolo_model.predict(
        source=annotated,
        verbose=False,
        device=yolo_device,
        conf=conf_threshold,
    )

    if results:
        result = results[0]
        _draw_yolo_detections(annotated, result)

        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                detected_objects.add(result.names.get(cls_id, str(cls_id)))

    return annotated, detected_objects


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
