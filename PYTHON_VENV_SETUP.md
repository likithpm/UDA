# Python Virtual Environment Setup (Windows)

This project uses a local virtual environment at `.venv`.

## 1. Create virtual environment

From the project root:

```powershell
python -m venv .venv
```

## 2. Activate virtual environment

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Command Prompt (cmd)

```bat
.venv\Scripts\activate.bat
```

## 3. Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## 5. Verify installation

```powershell
python -c "import torch, torchvision, cv2, streamlit, ultralytics, librosa, numpy, pandas, matplotlib, sklearn; print('All imports OK')"
```

## 6. Deactivate when done

```powershell
deactivate
```

---

## Installed dependencies in this project

- torch
- torchvision
- opencv-python
- streamlit
- ultralytics
- librosa
- numpy
- pandas
- matplotlib
- scikit-learn
