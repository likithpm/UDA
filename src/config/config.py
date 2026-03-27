"""Centralized project configuration using relative paths only."""

from pathlib import Path

# Base directory expressed relative to this file (no absolute paths).
BASE_DIR = Path(__file__).parent / ".." / ".."

# Existing dataset paths.
DATA_DIR = BASE_DIR / "data"
AQUATIC_DATA_DIR = DATA_DIR / "aquatic_animals_cleaned"
HUMAN_DATA_DIR = DATA_DIR / "human_cleaned"
WAVEFORM_DATA_DIR = DATA_DIR / "waveforms_cleaned"
YOLO_DATASET_DIR = BASE_DIR / "dataset_yolo"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
AUTO_LABEL_DATASET_DIR = BASE_DIR / "dataset_auto_labels"
AUTO_LABEL_DATA_YAML = AUTO_LABEL_DATASET_DIR / "data.yaml"

# Model artifact save paths.
MODELS_DIR = BASE_DIR / "models"
IMAGE_MODELS_DIR = MODELS_DIR / "image"
DETECTION_MODELS_DIR = MODELS_DIR / "detection"
AUDIO_MODELS_DIR = MODELS_DIR / "audio"
YOLO_WEIGHTS_PATH = DETECTION_MODELS_DIR / "yolo_weights.pt"

# Optional convenience map for downstream modules.
DATA_PATHS = {
    "aquatic": AQUATIC_DATA_DIR,
    "human": HUMAN_DATA_DIR,
    "waveform": WAVEFORM_DATA_DIR,
}

MODEL_PATHS = {
    "image": IMAGE_MODELS_DIR,
    "detection": DETECTION_MODELS_DIR,
    "audio": AUDIO_MODELS_DIR,
}
