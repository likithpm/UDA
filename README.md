# 🌊 Underwater Multimodal AI Detection System

A real-time multimodal AI project for underwater scene understanding that combines:

- Computer vision using Ultralytics YOLOv8 for marine object detection
- Audio classification using a PyTorch CNN on mel spectrograms
- Live visualization and analytics through a Streamlit dashboard

The system supports video file inference and webcam mode, with synchronized object and audio insights over time.

---

## ✨ Features

- 🐠 Object detection using YOLOv8
- 🔊 Audio classification across multiple underwater acoustic classes (about 49 classes)
- ⏱ Dynamic audio prediction over timeline chunks
- 🖥 Streamlit dashboard UI for interactive monitoring
- 🎥 Real-time video processing with annotated frames
- 🎯 Confidence filtering for detection stability
- 📥 Downloadable processed video output

---

## 🧠 Models Used

### A) YOLO Model (Object Detection)

- Framework: Ultralytics YOLOv8
- Target classes:
  - crab
  - dolphin
  - octopus
  - seahorse
  - seal
  - seaturtle
  - shark
  - starfish
- Typical training configuration:
  - epochs: about 50
  - image size: 640
  - batch size: configurable (commonly 8)
- Reported performance (project benchmark): mAP about 0.91

### B) Audio Model (Classification)

- Framework: PyTorch CNN
- Input representation:
  - mel spectrogram
  - sample rate: 22050
  - n_mels: 128
  - audio duration window: 3 seconds
- Architecture:
  - Conv2D blocks with ReLU + MaxPool
  - Fully connected head with Dropout
- Number of classes: about 49
- Typical validation accuracy: about 85 to 90 percent

---

## 🗂 Dataset Structure

```text
data/
├── aquatic_animals_cleaned/      # 8 marine image classes for detection
├── human_cleaned/                # present in project, ignored for detection pipeline
├── waveforms_cleaned/            # underwater/audio class folders
└── test_videos/                  # sample videos for inference testing
```

---

## 📦 Dataset & Models

Due to GitHub size limitations, datasets and trained models are not included in this repository.

They can be downloaded from the following link:

🔗 demo.googledrive

(Note: This is a placeholder link and will be updated with the actual dataset.)

---

## 🏗 Project Structure

```text
src/
├── app/          # Streamlit dashboard
├── config/       # central paths and settings
├── data/         # dataset loaders and conversion scripts
├── inference/    # video/audio/pipeline inference
├── models/       # model definitions (image, audio, yolo wrappers)
└── training/     # training scripts for image, audio, and yolo
```

Additional directories:

- models/: saved model weights and class mappings
- dataset_yolo/: generated YOLO training dataset

---

## ⚙️ Installation

### 1) Create virtual environment

```powershell
python -m venv .venv
```

### 2) Activate environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <repo_url>
cd <repo_name>
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

Windows:

```powershell
.\.venv\Scripts\Activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Instructions

### 1) Convert aquatic image dataset to YOLO format

```powershell
python src/data/convert_to_yolo.py
```

### 2) Train YOLO model

```powershell
python src/training/train_yolo.py
```

### 3) Train audio model

```powershell
python src/training/train_audio.py
```

---

## 🔍 Inference

### Video object detection

```powershell
python src/inference/predict_video.py --video data/test_videos/fish_video.mp4
```

### Multimodal pipeline (video + audio)

```powershell
python src/inference/pipeline.py --video data/test_videos/fish_video.mp4
```

---

## 🖥 Run Application

```powershell
streamlit run src/app/streamlit_app.py
```

The dashboard provides:

- Dynamic audio labels synchronized to video timeline
- Real-time object detections and insights
- Confidence threshold control
- Live webcam mode toggle
- Processed video playback and download

---

## 🚀 Run Project

### Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

### Run Video Pipeline

```bash
python src/inference/pipeline.py --video data/test_videos/sample.mp4
```

---

## 🔄 How It Works

1. Input video stream is read frame by frame.
2. YOLOv8 performs object detection on each frame.
3. Audio is extracted from video and split into time chunks.
4. Each chunk is converted to mel spectrogram and classified by the CNN.
5. Audio predictions are aligned to current video timestamp.
6. Streamlit displays synchronized audio + object insights in real time.

---

## 🚀 Key Highlights

- Multimodal AI design (vision + audio)
- Real-time synchronized inference
- Professional interactive dashboard
- Clean modular architecture for extensibility

---

## ⚠️ Limitations

- Bounding boxes are generated from available labels and may be approximate
- CPU-only execution can reduce FPS and responsiveness
- Dataset diversity and annotation quality directly impact generalization

---

## 🛠 Future Improvements

- Improve annotation quality and box precision
- Optimize GPU acceleration and model quantization
- Expand class coverage for both image and audio domains
- Deploy to cloud or edge runtime for production use

---

## 👤 Author

Developed as an underwater multimodal AI research and engineering project.

If you are extending this repository, consider adding:

- your name
- institution or team
- contact or portfolio link
