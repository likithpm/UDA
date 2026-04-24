# Underwater Multimodal AI Detection System

Real-time underwater monitoring system that combines YOLOv8 object detection, audio classification, threat intelligence, and a conversational assistant (SaiG) inside a single Streamlit dashboard.

---

## Overview

This project provides:

- Dynamic underwater object dataset auto-labeling
- YOLOv8 training with model reuse (skip retraining when weights already exist)
- Multimodal inference (video + audio)
- Threat scoring with temporal stability filtering
- Single-page Streamlit dashboard with live feed, insights, and SaiG chat
- OpenAI-compatible LLM integration for frame-aware assistant responses
- Intelligent fallback assistant mode when API keys are not configured

---

## Key Features

### Data and Labeling

- Automatic class discovery from dataset folders (no hardcoded class list)
- Data validation for malformed or duplicate class folder names
- Auto-generation of YOLO labels using pretrained YOLOv8
- Corrupted image handling during annotation
- Generated YOLO config at `dataset_auto_labels/data.yaml` with `nc` and ordered `names`
- Strict class consistency checks between folders, labels, and YAML

### Training

- YOLO training pipeline validates dataset before training
- Existing trained YOLO weights are reused by default
- Optional force retrain mode available
- Out-of-range / malformed label entries are sanitized before training

### Inference

- Inference class names are sourced from `dataset_auto_labels/data.yaml`
- Strict model-vs-YAML class consistency checks
- Out-of-range class IDs are skipped safely
- Performance controls for FPS throttling, frame skipping, and inference size

### Dashboard and UX

- Single-page dashboard flow:
	- Header
	- Controls panel
	- Input panel (upload / live camera)
	- Detection feed
	- Detection insights
	- SaiG assistant (quick analyze + full chat)
	- Threat history and analytics
	- Footer
- Premium Blue Eclipse theme
- Responsive layout for desktop/mobile

### SaiG Assistant

- One-click quick action: "Ask SaiG About This Frame"
- Full conversational follow-up chat
- Real LLM integration via OpenAI-compatible chat completions API
- Frame image + current detections + conversation context passed to model
- Intelligent fallback mode when API is unavailable or fails

---

## System Flow

Video stream -> YOLO detection -> detected objects -> threat logic -> alerts + logs  
Audio stream -> CNN audio classifier -> audio class  
Dashboard fusion -> live cards + history + analytics  
SaiG -> frame + detections + chat context -> analysis response

---

## Repository Structure

```text
src/
	app/           # Streamlit dashboard
	config/        # Project paths and settings
	data/          # Auto-labeling and conversion scripts
	inference/     # Inference and multimodal pipeline
	models/        # Model definitions (audio/image)
	training/      # Training scripts

data/
	aquatic_animals_cleaned/
	waveforms_cleaned/
	test_videos/

dataset_auto_labels/
	images/
	labels/
	data.yaml

models/
	detection/
	audio/
	image/
```

---

## Setup

1. Create a virtual environment.

```powershell
python -m venv .venv
```

2. Activate it in PowerShell.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

---

## Data Preparation

Generate pseudo labels and dataset YAML:

```powershell
python src/data/auto_annotate.py
```

This creates/refreshes:

- `dataset_auto_labels/images`
- `dataset_auto_labels/labels`
- `dataset_auto_labels/data.yaml`

---

## Training

Train YOLO (reuses existing model by default):

```powershell
python src/training/train_yolo.py
```

Force retraining from code entrypoint:

```powershell
python -c "from src.training.train_yolo import run_yolo_training; run_yolo_training(force_train=True)"
```

Train audio model:

```powershell
python src/training/train_audio.py
```

---

## Run Dashboard

```powershell
streamlit run src/app/streamlit_app.py
```

In the app:

1. Upload a video or enable live camera.
2. Click Start Detection.
3. Use controls (core + advanced settings) for performance/threat tuning.
4. Use "Ask SaiG About This Frame" for instant analysis.
5. Continue with follow-up chat prompts.

---

## SaiG API Configuration

SaiG supports OpenAI-compatible APIs using environment variables:

- `OPENAI_API_KEY` or `SAIG_API_KEY`
- `SAIG_API_BASE_URL` (default: `https://api.openai.com/v1/chat/completions`)
- `SAIG_MODEL` (default: `gpt-4o-mini`)

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your_key_here"
$env:SAIG_MODEL = "gpt-4o-mini"
streamlit run src/app/streamlit_app.py
```

If no API key is set, SaiG automatically uses intelligent fallback mode.

---

## Threat Logic

- Confidence-weighted object aggregation
- 5-second rolling history window
- Stability duration threshold to reduce flicker
- Threat levels (low / medium / high)
- Optional sound alerts for high stable threats

Default high-risk examples include shark and submarine classes.

---

## Notes

- Keep `dataset_auto_labels/data.yaml` aligned with class folders in `data/aquatic_animals_cleaned`.
- Inference labels are resolved from YAML class order, not hardcoded names.
- The dashboard includes startup validation and safe-stop behavior on critical config/model mismatch.
