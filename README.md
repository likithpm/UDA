# 🌊 Underwater Multimodal AI Detection System

A real-time AI system that detects underwater objects and analyzes audio using computer vision and deep learning, with an intelligent threat alert system.

---

## 🚀 Features

- Real-time object detection using YOLOv8
- Audio classification using CNN (mel spectrograms)
- Multimodal detection (video + audio)
- Confidence-weighted threat detection
- Temporal stability filtering (anti-flicker alerts)
- Dynamic threat levels (1-10)
- Sound alert system (toggle-based)
- Threat history logging
- Analytics dashboard (charts)
- Ocean-themed UI with glassmorphism design

---

## 🧠 System Architecture

Video Input -> YOLO Detection -> Object Labels  
Audio Input -> CNN Model -> Audio Labels  
Downstream fusion:  
Temporal Aggregation -> Confidence Scoring -> Stability Filter  
Then:  
Threat Detection -> UI Alerts + Logging

---

## 📂 Dataset

```text
data/
├── aquatic_animals_cleaned/   # 10 image classes
├── waveforms_cleaned/         # underwater audio dataset
└── test_videos/               # inference/demo videos
```

Detection classes:

- crab
- dolphin
- human
- octopus
- seahorse
- seal
- seaturtle
- shark
- starfish
- submarine

Notes:

- Images are auto-annotated using YOLO (pseudo-labeling).
- Bounding boxes are generated automatically.

---

## 🔄 Auto-Annotation Pipeline

- Uses pretrained YOLO to generate bounding boxes
- Confidence threshold tuned (about 0.3) for better coverage
- Iterative self-training supported

Note:

This replaces manual labeling and improves localization over time.

---

## 🤖 Models

### YOLOv8

- Used for object detection
- Pretrained model: `yolov8n`
- Fine-tuned on custom underwater dataset

### CNN (Audio Model)

- Input: Mel spectrogram
- Output: Audio class prediction
- Typical validation accuracy: about 85% to 95%

---

## 🚨 Threat Detection System

- Uses object detection results
- Applies confidence-weighted scoring
- Uses a 5-second rolling window
- Uses stability threshold (user-controlled)
- Assigns threat levels (1-10)

Examples:

- Shark -> High threat
- Submarine -> High threat
- Seahorse -> Low threat

---

## 🎨 UI Dashboard

- Ocean-themed design
- Glassmorphism cards
- Real-time video feed
- Alerts with color coding
- Analytics charts
- Threat history log

---

## 🗂 Project Structure

```text
src/
├── app/         # streamlit dashboard
├── config/      # paths and settings
├── data/        # conversion and annotation scripts
├── inference/   # multimodal inference pipeline
├── models/      # model definitions
└── training/    # training scripts

models/
├── detection/
├── audio/
└── image/

dataset_auto_labels/
├── images/
├── labels/
└── data.yaml
```

---

## ⚙️ Installation

1. Create virtual environment:

```powershell
python -m venv .venv
```

2. Activate environment (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 🏋️ Training

1. Regenerate auto labels:

```powershell
python src/data/auto_annotate.py
```

2. Train YOLO model:

```powershell
python src/training/train_yolo.py
```

3. Train audio model:

```powershell
python src/training/train_audio.py
```

---

## ▶️ How to Run

1. Activate environment.
2. Run Streamlit app:

```powershell
streamlit run src/app/streamlit_app.py
```

---

## ⚠️ Dataset Note

Dataset is not included due to size.

A downloadable link will be provided:

[Dataset Link - Coming Soon]

---

## 📸 Demo

![Demo](assets/demo.png)

(Optional: add video later)

---

## 🔮 Future Improvements

- Manual annotation for higher accuracy
- Deployment to cloud
- Real-time underwater drone integration
- Advanced analytics
