# ✋ Sign Language Recognition

A deep learning system that identifies **American Sign Language (ASL) hand signs (A–Z)**
in real time — from a webcam or uploaded image — built with TensorFlow, MobileNetV2, and Gradio.

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | **99.74 %** |
| Classes | 26 (A – Z) |
| Model | MobileNetV2 (fine-tuned) |

## Project Structure
```
sign-language-recognition-/
├── src/
│   ├── config.py        ← paths + hyperparameters
│   ├── preprocess.py    ← data loading + augmentation
│   ├── model.py         ← MobileNetV2 CNN architecture
│   ├── train.py         ← two-phase training pipeline
│   └── predict.py       ← inference logic
├── app/
│   └── app.py           ← Gradio UI (webcam + upload)
├── tests/
│   └── test_predict.py
├── models/
│   ├── training_curves_phase1.png
│   └── training_curves_phase2.png
├── requirements.txt
└── README.md
```
## Setup

```bash
git clone https://github.com/meghanaguddanti/sign-language-recognition-.git
cd sign-language-recognition-
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

| Split | Link |
|-------|------|
| Train | https://drive.google.com/drive/folders/1QC9e438FpPH0fxS1UusfEj07gs4IOkHm |
| Test  | https://drive.google.com/drive/folders/1BzkKVI2S3oWmXAa6m9TgtZ1DkxGl2z82 |

Place in `data/raw/train/` and `data/raw/test/` with one subfolder per class (A–Z).

## Train

```bash
cd src
python train.py
```

## Run App

```bash
cd app
python app.py
```

Open **http://127.0.0.1:7860**

## Tech Stack

Python 3.11 · TensorFlow 2.x · MobileNetV2 · Gradio · OpenCV · MediaPipe · scikit-learn

## Author

**Meghana Guddanti** — [github.com/meghanaguddanti](https://github.com/meghanaguddanti)
