# Automated Analysis of Finger Tapping in Parkinson’s Disease Using Computer Vision and AI

## Overview
This project presents an AI-based system for analysing finger tapping behaviour in Parkinson’s disease using computer vision and signal analysis.

The project uses two complementary approaches:

- **YOLO** for saved-video finger detection and offline analysis
- **MediaPipe** for live hand landmark tracking during real-time demonstration

The system detects finger movement, converts motion into measurable signals, extracts movement features, and supports an estimated UPDRS-style severity output for demonstration purposes.


---

## Project Objectives
The main objectives of the project are:

- detect finger motion from video
- analyse thumb and index finger movement
- convert movement into a time-series signal
- extract meaningful movement features
- compare Parkinson’s and control patterns
- support both saved-video and live demonstration workflows

---

## Team Members
- **Aatma Ram**  Project Manager / Scrum Master
- **Eknoor Sidhu**  AI & Computer Vision Engineer
- **Kartik Chauhan**  Data & Software Engineer

---

## Methods Used

### 1. YOLO-based saved-video analysis
A custom YOLO model is used to detect the **thumb** and **index finger** in saved finger tapping videos. Their positions are used to calculate the Euclidean distance frame by frame, generating a movement signal over time.

### 2. MediaPipe-based live demo
MediaPipe hand landmark tracking is used as a supplementary live-demo method for robust real-time finger tracking and movement estimation.

### 3. Signal analysis
The distance between the thumb and index finger is used to form a time-series signal. From this signal, features such as amplitude, speed, variability, and tap count can be derived.

---

## Mathematical Basis
The distance between the thumb and index finger is computed using the Euclidean distance formula:

**d = √((x₂ − x₁)² + (y₂ − y₁)²)**

This distance is tracked over time to represent tapping motion.

---

## Expected Project Structure

```text
PDA/
├── PDAV/                                  # Saved input videos
├── frames_raw/                            # Extracted raw frames for annotation
├── DATA/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml                          # YOLO dataset configuration
├── runs/
│   ├── finger_detector/
│   │   └── weights/
│   │       └── best.pt                    # Original trained YOLO model
│   └── finger_detector_retrain/
│       └── weights/
│           └── best.pt                    # Retrained YOLO model (if used)
├── signal_output/                         # CSV files, graphs, and extracted outputs
├── extract_frames.py                      # Extracts frames from saved videos
├── prepare_yolo_dataset.py                # Splits labeled data into train/valid
├── train_yolo.py                          # Retrains YOLO on updated data
├── video_test_yolo.py                     # Runs YOLO on a saved video
├── mediapip.py                            # Runs MediaPipe-based live demo
├── README.md
└── requirements.txt
```

---

## Requirements

### Software
- Python 3.10–3.12
- Windows 10/11
- VS Code or terminal
- Webcam (only for live MediaPipe demo)

### Python libraries
- ultralytics
- opencv-python
- numpy
- matplotlib
- pandas
- mediapipe
- openpyxl (optional)

---

## Installation

Install required packages:

```bash
pip install ultralytics opencv-python numpy matplotlib pandas mediapipe openpyxl
```

If using the MediaPipe setup, use compatible versions such as:

```bash
pip install numpy==1.26.4 mediapipe==0.10.21 opencv-python==4.11.0.86
```

---

## Dataset Format for YOLO
YOLO training requires a custom dataset with:

- image files in `train/images` and `valid/images`
- matching label files in `train/labels` and `valid/labels`
- a `data.yaml` file describing the dataset

### Class labels
```yaml
names:
  0: thumb
  1: index_finger
```

---

## How to Run the Project

## 1. Extract frames from saved videos
Use this if you want to prepare additional images for annotation and retraining:

```bash
python extract_frames.py
```

This script extracts frames from videos inside the `PDAV` folder.


---

## 2. Train or retrain YOLO
To train using your updated dataset:

```bash
python train_yolo.py
```

This uses the dataset defined in `DATA/data.yaml`.

---

## 3. Run YOLO on a saved video
To test the saved-video workflow:

```bash
python video_test.py
```

This should:
- open a saved video
- run YOLO frame by frame
- show thumb/index detections
- estimate movement behaviour
- display basic output such as distance, tap count, and score

---

## 5. Run MediaPipe live demo
To test the real-time demonstration:

```bash
python mediapip.py
```

This should:
- open the webcam or selected video source
- track thumb and index landmarks
- compute finger distance
- estimate live movement behaviour

---

## Input and Output

### Inputs
- finger tapping saved videos
- live webcam feed (for MediaPipe demo)
- trained YOLO weights
- annotated dataset for retraining

### Outputs
- YOLO-detected saved-video frames
- MediaPipe live tracking demo
- movement signal graphs
- extracted movement features
- estimated UPDRS-style scores
- comparison between Parkinson’s and control subjects

---

## Saved-Video vs Live-Demo Roles

### YOLO
Used as the **primary custom model** for:
- saved-video detection
- signal generation
- feature extraction
- comparison of Parkinson’s and control samples

### MediaPipe
Used as a **supplementary live-demo model** for:
- robust real-time hand landmark tracking
- live distance measurement
- stable demo presentation

---

## Troubleshooting

### 1. Video opens webcam instead of saved file
Check that your script uses a **file path string** instead of `0`.

Correct example:
```python
VIDEO_SOURCE = r"D:\PDA\PDAV\C30.MOV"
```

Wrong example:
```python
VIDEO_SOURCE = 0
```

---



## References
- Ultralytics. YOLO Documentation.
- Google. MediaPipe Hands Documentation.
- Movement Disorder Society. MDS-UPDRS.
- OpenCV Documentation.
- Zhang et al. Finger Tapping Analysis for Parkinson’s Disease.

---

