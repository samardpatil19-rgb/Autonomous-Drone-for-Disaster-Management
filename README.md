# Autonomous Drone for Disaster Management

An AI-powered autonomous drone system for detecting survivors in disaster-struck environments. The system uses a YOLOv8n deep learning model trained on the SARD (Search and Rescue Dataset) for real-time person detection from aerial imagery, deployed on a Raspberry Pi 5 companion computer with a Pixhawk 2.4.8 flight controller.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training Pipeline](#training-pipeline)
  - [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [Hardware Setup](#hardware-setup)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

This project implements an end-to-end autonomous drone-based survivor detection system designed for disaster management scenarios. The drone autonomously follows pre-planned waypoints using Mission Planner, continuously captures aerial imagery, and uses an onboard machine learning model to detect people on the ground. Upon detection, the drone hovers over the identified location and transmits GPS coordinates to ground station authorities for rescue operations.

### Key Features

- **Real-time person detection** using YOLOv8n optimized for edge deployment
- **Autonomous flight** via waypoint-based missions in Mission Planner
- **Automatic hover-on-detection** with GPS coordinate reporting
- **Consecutive detection filtering** to minimize false positives
- **CSV-based detection logging** with timestamps and GPS coordinates
- **Modular architecture** separating detection, flight control, and reporting

---

## System Architecture

```
+---------------------------------------------+
|                   DRONE                      |
|                                              |
|   +------------+      +------------------+  |
|   | Pixhawk    |<---->| Raspberry Pi 5   |  |
|   | 2.4.8 (FC) | UART | (Companion PC)   |  |
|   +------------+      +--------+---------+  |
|                                 | CSI        |
|                        +--------+---------+  |
|                        | Pi Camera        |  |
|                        | Module 3         |  |
|                        +------------------+  |
+---------------------------------------------+
         ^ Telemetry Radio (915 MHz)
         v
+--------------------+
|   Ground Station   |
|  (Mission Planner) |
+--------------------+
```

---

## Project Structure

```
Autonomous-Drone-for-Disaster-Management/
|-- README.md
|-- requirements.txt                     # Training dependencies (laptop)
|-- data.yaml                            # YOLO dataset configuration
|-- config/
|   |-- __init__.py
|   +-- training_config.py               # Centralized training hyperparameters
|-- scripts/
|   |-- 01_download_dataset.py           # Download SARD dataset from Kaggle
|   |-- 02_prepare_dataset.py            # Validate dataset and create config
|   |-- 03_train_model.py                # Fine-tune YOLOv8n on SARD
|   |-- 04_evaluate_model.py             # Evaluate model metrics (mAP, P, R)
|   |-- 05_export_model.py               # Export to NCNN/ONNX for Pi 5
|   +-- 06_test_webcam.py                # Real-time webcam detection test
|-- pi_deployment/
|   |-- requirements_pi.txt              # Pi-specific dependencies
|   |-- config.py                        # Hardware and detection settings
|   |-- detector.py                      # YOLOv8 inference module
|   |-- drone_controller.py              # DroneKit + Pixhawk interface
|   |-- gps_reporter.py                  # GPS logging and alerting
|   +-- main.py                          # Main entry point for Pi
+-- docs/
    +-- setup_guide.md                   # Detailed setup instructions
```

---

## Tech Stack

| Component            | Technology                          |
|----------------------|-------------------------------------|
| ML Model             | YOLOv8n (Ultralytics)               |
| Training Framework   | PyTorch with CUDA                   |
| Dataset              | SARD (Search and Rescue Dataset)    |
| Edge Inference       | NCNN / ONNX Runtime                 |
| Flight Controller    | Pixhawk 2.4.8 with ArduPilot        |
| Companion Computer   | Raspberry Pi 5 (8GB)                |
| Camera               | Pi Camera Module 3                  |
| Drone Communication  | DroneKit + MAVLink                  |
| Ground Station       | Mission Planner                     |

---

## Getting Started

### Prerequisites

**For Training (Laptop/PC):**
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (tested on RTX 3050)
- Kaggle account with API credentials

**For Deployment (Raspberry Pi 5):**
- Raspberry Pi 5 (4GB or 8GB)
- Raspberry Pi OS 64-bit (Bookworm)
- Pi Camera Module 3 or USB webcam
- Pixhawk 2.4.8 flight controller

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samardpatil19-rgb/Autonomous-Drone-for-Disaster-Management.git
   cd Autonomous-Drone-for-Disaster-Management
   ```

2. **Install training dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA** (if not already installed):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Configure Kaggle API credentials:**
   - Download `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings)
   - Place it in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### Training Pipeline

Execute the following scripts sequentially:

```bash
# Step 1: Download the SARD dataset
python scripts/01_download_dataset.py

# Step 2: Validate and prepare the dataset
python scripts/02_prepare_dataset.py

# Step 3: Train YOLOv8n (approx. 1-2 hours on RTX 3050)
python scripts/03_train_model.py

# Step 4: Evaluate model performance
python scripts/04_evaluate_model.py

# Step 5: Export model for Raspberry Pi 5
python scripts/05_export_model.py

# Step 6: Test detection using laptop webcam
python scripts/06_test_webcam.py
```

### Raspberry Pi Deployment

1. **Install dependencies on the Pi:**
   ```bash
   pip install -r pi_deployment/requirements_pi.txt
   sudo apt install python3-picamera2
   ```

2. **Transfer the trained model:**
   ```bash
   scp -r runs/detect/disaster_rescue_detector/weights/best.pt pi@<PI_IP>:~/drone/model/
   ```

3. **Copy deployment files:**
   ```bash
   scp -r pi_deployment/* pi@<PI_IP>:~/drone/
   ```

4. **Run the detection system:**
   ```bash
   # Camera test only (no drone connection)
   python main.py --test-camera

   # Full autonomous mode
   python main.py
   ```

---

## Hardware Setup

### Wiring: Raspberry Pi 5 to Pixhawk 2.4.8 (TELEM2)

| Raspberry Pi 5     | Pixhawk 2.4.8 (TELEM2) |
|---------------------|-------------------------|
| GPIO 14 (TX)        | Pin 3 (RX)              |
| GPIO 15 (RX)        | Pin 2 (TX)              |
| GND                 | Pin 6 (GND)             |

> **Important:** Do not connect 5V from Pixhawk to the Pi. Power each device independently.

### TELEM2 Pin Reference

| Pin | Function                      |
|-----|-------------------------------|
| 1   | +5V (do not connect to Pi)    |
| 2   | TX                            |
| 3   | RX                            |
| 4   | CTS (unused)                  |
| 5   | RTS (unused)                  |
| 6   | GND                           |

---

## Configuration

### Training Configuration (`config/training_config.py`)

| Parameter            | Default | Description                        |
|----------------------|---------|------------------------------------|
| `MODEL_VARIANT`      | yolov8n | YOLOv8 Nano variant                |
| `EPOCHS`             | 50      | Number of training epochs           |
| `BATCH_SIZE`         | 4       | Training batch size                 |
| `IMAGE_SIZE`         | 640     | Input image resolution              |
| `CONFIDENCE_THRESHOLD` | 0.5  | Minimum detection confidence        |

### Pi Deployment Configuration (`pi_deployment/config.py`)

| Parameter              | Default       | Description                      |
|------------------------|---------------|----------------------------------|
| `PIXHAWK_CONNECTION`   | /dev/serial0  | UART serial port                 |
| `PIXHAWK_BAUD`         | 57600         | MAVLink baud rate                |
| `HOVER_DURATION`       | 15s           | Hover time after detection       |
| `MIN_DETECTION_COUNT`  | 2             | Consecutive frames to confirm    |
| `DETECTION_INTERVAL`   | 0.5s          | Time between inference frames    |

---

## How It Works

1. **Mission Planning:** Waypoints are configured in Mission Planner at 10-15m altitude.
2. **Autonomous Flight:** The drone follows the pre-planned waypoint mission in AUTO mode.
3. **Continuous Detection:** The Pi Camera captures frames and the YOLOv8n model runs inference on each frame to detect persons.
4. **Detection Confirmation:** A minimum of two consecutive positive detections is required to trigger an alert, reducing false positives.
5. **Hover and Report:** Upon confirmed detection, the drone switches to LOITER mode (hover), logs the GPS coordinates with a timestamp to a CSV file, and alerts the ground station.
6. **Mission Resumption:** After a configurable hover duration, the drone resumes the waypoint mission in AUTO mode.

---

## License

This project is developed as part of a Final Year academic project for educational and research purposes.
