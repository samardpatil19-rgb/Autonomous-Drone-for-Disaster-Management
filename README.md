# 🚁 Autonomous Drone for Disaster Management

An AI-powered autonomous drone system for detecting survivors in disaster-struck environments. Uses a **YOLOv8s** deep learning model fine-tuned on the **SARD + VisDrone** datasets, deployed on a **Raspberry Pi 5** with a **Pixhawk 2.4.8** flight controller. Features real-time person detection, autonomous hover-on-detection, live video streaming, and dual-channel alerting.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Hardware Setup](#hardware-setup)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Test Results](#test-results)
- [License](#license)

---

## Overview

This project implements an end-to-end autonomous drone-based survivor detection system designed for disaster management scenarios (earthquakes, floods, landslides). The drone autonomously follows pre-planned waypoints using Mission Planner, continuously captures aerial imagery via onboard camera, and runs real-time ML inference to detect people on the ground. Upon confirmed detection, the drone hovers over the location, streams live video, and transmits GPS coordinates to ground station authorities.

### Key Features

- **Real-time person detection** using YOLOv8s (75.3% precision, 69.6% mAP@50)
- **Threaded architecture** — smooth 15 FPS video stream while YOLO runs in background
- **Autonomous hover-on-detection** with GPS coordinate reporting
- **Live video dashboard** accessible via browser at `http://<Pi_IP>:5000`
- **Dual-channel alerting:**
  - MAVLink STATUSTEXT via telemetry radio (works without internet)
  - HTTP POST with GPS + screenshot over 4G/WiFi
- **Consecutive detection filtering** (2+ frames) to minimize false positives
- **Edge AI** — all inference runs locally on Pi (no cloud dependency)
- **NCNN optimized** model for ARM processors
- **Detection logging** — saves annotated screenshots + GPS to CSV

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│                    DRONE                         │
│                                                  │
│  ┌──────────┐    UART     ┌──────────────────┐  │
│  │ Pixhawk  │◄───────────►│  Raspberry Pi 5  │  │
│  │  2.4.8   │  (MAVLink)  │  + Camera Mod 3  │  │
│  │  (FC)    │             │  + YOLO Model     │  │
│  └────┬─────┘             └────────┬─────────┘  │
│       │                            │             │
│  ┌────┴─────┐              ┌───────┴────────┐   │
│  │ 4x BLDC  │              │  4G LTE Dongle │   │
│  │ Motors   │              │  (WiFi/USB)    │   │
│  │ SimonK   │              └───────┬────────┘   │
│  │ 30A ESC  │                      │             │
│  └──────────┘                      │ WiFi/4G     │
└────────────────────────────────────┼─────────────┘
                                     │
                              ┌──────┴──────┐
                              │   Laptop    │
                              │ (Ground     │
                              │  Station)   │
                              │ - Live Feed │
                              │ - Mission   │
                              │   Planner   │
                              └─────────────┘
```

---

## Model Performance

| Parameter | Value |
|---|---|
| **Architecture** | YOLOv8s (Small variant) |
| **Base Model** | Pre-trained on COCO (80 classes) |
| **Fine-tuned on** | SARD + VisDrone (aerial person imagery) |
| **Task** | Single-class detection (Person) |
| **Input Resolution** | 1280 × 1280 px |
| **Training Epochs** | 100 (full convergence) |
| **Batch Size** | 2 (RTX 3050, 4GB VRAM) |
| **Backbone Freeze** | First 10 layers (retains COCO features) |
| **Training GPU** | NVIDIA RTX 3050 (4GB VRAM) |
| **Edge Deployment** | NCNN format (ARM-optimized) |

### Best Model Metrics

| Metric | Score |
|---|---|
| **Precision** | **99.01%** |
| **Recall** | **69.7%** |
| **mAP@50** | **69.6%** |
| **mAP@50-95** | **37.9%** |

### Inference Speed

| Platform | Format | Speed |
|---|---|---|
| RTX 3050 (Laptop) | PyTorch (.pt) | ~5ms/frame |
| Raspberry Pi 5 | NCNN | ~200-500ms/frame |

### Training Augmentations

| Augmentation | Value | Purpose |
|---|---|---|
| Vertical Flip | 50% | Drone can view from any angle |
| Horizontal Flip | 50% | Direction invariance |
| Mosaic | 100% | Context diversity |
| Scale | ±50% | Different altitudes |
| HSV Jitter | H:0.015, S:0.7, V:0.4 | Lighting robustness |
| Random Erasing | 40% | Occlusion handling |

---

## Dataset

The model was trained on a combined dataset of **14,384 aerial images**:

| Split | SARD | VisDrone | Combined |
|---|---|---|---|
| **Train** | 4,041 | 6,471 | **10,512** |
| **Validation** | 1,144 | 548 | **1,692** |
| **Test** | 570 | 1,610 | **2,180** |
| **Total** | **5,755** | **8,629** | **14,384** |

- **SARD** (Search And Rescue Dataset): Aerial images of people in wilderness/disaster scenarios
- **VisDrone** (VisDrone2019-DET): Drone-perspective images, filtered to pedestrian/people classes only

---

## Project Structure

```
Autonomous-Drone-for-Disaster-Management/
├── README.md
├── requirements.txt                         # Training dependencies (laptop)
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── training_config.py                   # Training hyperparameters
│
├── scripts/
│   ├── 01_download_dataset.py               # Download SARD from Kaggle
│   ├── 01b_download_visdrone_merge.py        # Download VisDrone + merge
│   ├── 02_prepare_dataset.py                # Validate dataset
│   ├── 03_train_model.py                    # Fine-tune YOLOv8s (100 epochs)
│   ├── 04_evaluate_model.py                 # Evaluate model metrics
│   ├── 05_export_model.py                   # Export to NCNN/ONNX for Pi
│   └── 06_test_webcam.py                    # Real-time webcam test
│
├── pi_deployment/
│   ├── requirements_pi.txt                  # Pi dependencies
│   ├── config.py                            # Hardware & detection settings
│   ├── main.py                              # Main entry (threaded detection)
│   ├── detector.py                          # YOLO inference wrapper
│   ├── drone_controller.py                  # DroneKit Pixhawk interface
│   ├── communications.py                    # Dual-channel alerts (MAVLink + HTTP)
│   ├── video_stream.py                      # Flask MJPEG live streaming
│   └── gps_reporter.py                      # GPS logging utility
│
├── runs/detect/
│   ├── disaster_rescue_detector_v2/         # Training results
│   │   ├── weights/
│   │   │   ├── best.pt                      # Best model weights
│   │   │   ├── best.onnx                    # ONNX export
│   │   │   └── best_ncnn_model/             # NCNN for Raspberry Pi
│   │   ├── results.csv                      # Epoch-by-epoch metrics
│   │   ├── results.png                      # Training curves
│   │   ├── confusion_matrix.png             # Confusion matrix
│   │   ├── BoxF1_curve.png                  # F1 vs confidence
│   │   ├── BoxPR_curve.png                  # Precision-Recall curve
│   │   ├── val_batch*_pred.jpg              # Validation predictions
│   │   └── train_batch*.jpg                 # Training batch samples
│   └── predict2/                            # Test prediction results
│
├── Test images Phone/                       # Raw test images from drone
│   └── Drone_Images_for matching/
│
└── docs/
    └── setup_guide.md                       # Detailed setup instructions
```

---

## Tech Stack

### Hardware

| Component | Model | Purpose |
|---|---|---|
| Flight Controller | Pixhawk 2.4.8 | Autopilot, GPS navigation, stabilization |
| Companion Computer | Raspberry Pi 5 | AI inference, video streaming |
| Camera | Pi Camera Module 3 (IMX708) | 12MP, 4608×2592 |
| Motors | BLDC × 4 | Propulsion |
| ESCs | SimonK 30A × 4 | Motor speed control (PWM) |
| Telemetry | 433/915MHz Radio | MAVLink to Mission Planner |
| GPS | M8N + Compass | Navigation and positioning |
| Connectivity | 4G LTE WiFi Dongle | Remote internet for live feed |
| Power | 5V/5A UBEC + LiPo Battery | Powers Pi from drone battery |

### Software (Raspberry Pi)

| Library | Purpose |
|---|---|
| Ultralytics ≥8.1.0 | YOLOv8 inference engine |
| OpenCV ≥4.8.0 | Image processing |
| Flask ≥3.0.0 | MJPEG live video streaming |
| DroneKit ≥2.9.2 | MAVLink communication |
| PyMAVLink ≥2.4.0 | Low-level MAVLink protocol |
| Picamera2 | Pi Camera driver |
| NCNN | ARM-optimized neural network inference |
| Requests ≥2.31.0 | HTTP alerts |

### Software (Training/Laptop)

| Software | Purpose |
|---|---|
| PyTorch + CUDA | GPU-accelerated training |
| Ultralytics | YOLOv8 training framework |
| Mission Planner | Flight planning and monitoring |

### Protocols

| Protocol | Usage |
|---|---|
| MAVLink | Pi ↔ Pixhawk (UART, 57600 baud) |
| MJPEG/HTTP | Live video stream (Port 5000) |
| SSH | Remote Pi access |
| WiFi/4G | Data link |

---

## Getting Started

### Prerequisites

**For Training (Laptop):**
- Python 3.9+
- NVIDIA GPU with CUDA (tested on RTX 3050)
- Kaggle account with API credentials

**For Deployment (Pi 5):**
- Raspberry Pi 5 with Raspberry Pi OS 64-bit (Bookworm)
- Pi Camera Module 3
- Pixhawk 2.4.8 flight controller

### Installation

```bash
git clone https://github.com/yourusername/Autonomous-Drone-for-Disaster-Management.git
cd Autonomous-Drone-for-Disaster-Management
pip install -r requirements.txt
```

### Training Pipeline

```bash
python scripts/01_download_dataset.py           # Download SARD
python scripts/01b_download_visdrone_merge.py    # Download VisDrone + merge
python scripts/02_prepare_dataset.py             # Validate dataset
python scripts/03_train_model.py                 # Train YOLOv8s (~40 hours)
python scripts/04_evaluate_model.py              # Evaluate metrics
python scripts/05_export_model.py                # Export to NCNN/ONNX
python scripts/06_test_webcam.py                 # Test on webcam
```

### Raspberry Pi Deployment

```bash
# On Pi: Install dependencies
pip install -r pi_deployment/requirements_pi.txt
sudo apt install python3-picamera2

# From laptop: Transfer model and code
scp -r pi_deployment/ pi@<PI_IP>:~/drone_project/
scp -r runs/detect/disaster_rescue_detector_v2/weights/best_ncnn_model pi@<PI_IP>:~/drone_project/model/

# On Pi: Run
python main.py --test-camera    # Camera test (no drone)
python main.py                  # Full autonomous mode
```

---

## Hardware Setup

### Wiring: Raspberry Pi 5 ↔ Pixhawk 2.4.8 (TELEM2)

| Raspberry Pi 5 | Pixhawk TELEM2 |
|---|---|
| GPIO 14 / Pin 8 (TX) | Pin 3 (RX) |
| GPIO 15 / Pin 10 (RX) | Pin 2 (TX) |
| GND / Pin 6 | Pin 6 (GND) |

> **⚠️ Do NOT connect 5V from Pixhawk to Pi.** Power each device independently.

### Power: Drone Battery → Pi

```
[LiPo Battery] → [5V/5A UBEC] → [Cut USB-C Cable] → [Pi USB-C Port]
```

---

## Configuration

### Training (`config/training_config.py`)

| Parameter | Value | Description |
|---|---|---|
| `MODEL_VARIANT` | `yolov8s.pt` | YOLOv8 Small |
| `EPOCHS` | 100 | Full convergence |
| `BATCH_SIZE` | 2 | RTX 3050 (4GB VRAM) |
| `IMAGE_SIZE` | 1280 | High-res for small objects |
| `FREEZE_LAYERS` | 10 | Retain COCO backbone |

### Pi Deployment (`pi_deployment/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `PIXHAWK_CONNECTION` | `/dev/serial0` | UART serial port |
| `PIXHAWK_BAUD` | 57600 | MAVLink baud rate |
| `CAMERA_RESOLUTION` | (640, 480) | Inference resolution |
| `CONFIDENCE_THRESHOLD` | 0.5 | Min detection confidence |
| `HOVER_DURATION` | 15s | Hover time after detection |
| `MIN_DETECTION_COUNT` | 2 | Consecutive frames to confirm |
| `DETECTION_INTERVAL` | 0.5s | Time between YOLO runs |
| `STREAM_PORT` | 5000 | Live video feed port |

---

## How It Works

1. **Mission Planning:** Waypoints configured in Mission Planner at 10-15m altitude.
2. **Launch:** Drone takes off, operator switches to AUTO mode.
3. **Autonomous Search:** Pixhawk flies the waypoint path. Pi camera captures frames continuously.
4. **Detection:** YOLO runs in a background thread. Main thread streams smooth video at ~15 FPS.
5. **Confirmation:** 2+ consecutive positive detections required to trigger alert (reduces false positives).
6. **Hover & Alert:** Drone switches to LOITER mode. GPS coordinates + annotated screenshot sent via:
   - MAVLink STATUSTEXT → Mission Planner (via telemetry radio)
   - HTTP POST → Ground station (via 4G/WiFi)
   - Local save → SD card (timestamped images + CSV log)
7. **Resume:** After 15s hover + 5s cooldown, drone resumes AUTO mission.
8. **Live Monitoring:** Operator watches everything at `http://<Pi_IP>:5000`.

---

## Test Results

Test predictions on real drone imagery are available in `runs/detect/predict2/`. Training curves, validation images, confusion matrices, and PR curves are in `runs/detect/disaster_rescue_detector_v2/`.

---

## License

This project is developed as part of a Final Year academic project for educational and research purposes.
