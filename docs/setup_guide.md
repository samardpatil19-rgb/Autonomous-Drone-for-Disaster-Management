# Setup Guide — Drone Final Year Project

## Part 1: Laptop Setup (Training)

### Prerequisites
- Python 3.9+ installed
- RTX 3050 with CUDA drivers
- Kaggle account (for dataset download)

### Step 1: Install Dependencies
```bash
cd "C:\Projects\Drone Final Year Project"
pip install -r requirements.txt
```

### Step 2: Setup Kaggle API
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Click **"Create New Token"** — downloads `kaggle.json`
3. Place `kaggle.json` in `C:\Users\YourName\.kaggle\`
4. Alternatively, `kagglehub` will prompt you to log in on first use

### Step 3: Run Training Pipeline
```bash
# Run these in order:
python scripts/01_download_dataset.py    # ~5 minutes
python scripts/02_prepare_dataset.py     # ~30 seconds
python scripts/03_train_model.py         # ~1-2 hours
python scripts/04_evaluate_model.py      # ~5 minutes
python scripts/05_export_model.py        # ~2 minutes
python scripts/06_test_webcam.py         # interactive test
```

---

## Part 2: Raspberry Pi 5 Setup

### Prerequisites
- Raspberry Pi 5 (4GB or 8GB)
- Raspberry Pi OS (64-bit Bookworm)
- Pi Camera Module 3 (or USB webcam)
- MicroSD card (32GB+ recommended)

### Step 1: Update Pi
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Python Dependencies
```bash
cd ~/drone_project
pip install -r requirements_pi.txt
```

### Step 3: Install Pi Camera (if using Pi Camera Module)
```bash
sudo apt install python3-picamera2 -y
```

### Step 4: Enable UART for Pixhawk
```bash
sudo raspi-config
# → Interface Options → Serial Port
# → Login shell over serial: NO
# → Serial port hardware enabled: YES
# Reboot after
```

### Step 5: Copy Model from Laptop
Transfer the exported model from your laptop to the Pi:
```bash
# On laptop (using scp):
scp -r "runs/detect/disaster_rescue_detector/weights/best_ncnn_model" pi@<PI_IP>:~/drone_project/model/
```

### Step 6: Run Detection System
```bash
# Test camera only (no drone):
python main.py --test-camera

# Full mode (with Pixhawk connected):
python main.py
```

---

## Part 3: Pixhawk 2.4.8 Wiring

### Pi 5 GPIO → Pixhawk TELEM2

```
Raspberry Pi 5          Pixhawk 2.4.8 (TELEM2)
──────────────          ─────────────────────────
GPIO 14 (TX)  ───────►  Pin 3 (RX)
GPIO 15 (RX)  ◄───────  Pin 2 (TX)
GND           ◄───────► Pin 6 (GND)
```

> ⚠️ **IMPORTANT**: Do NOT connect 5V from Pixhawk to Pi — power them separately!

### TELEM2 Pin Layout (Pixhawk 2.4.8)
| Pin | Function |
|-----|----------|
| 1   | +5V (DO NOT CONNECT TO PI) |
| 2   | TX |
| 3   | RX |
| 4   | CTS (not used) |
| 5   | RTS (not used) |
| 6   | GND |

---

## Part 4: Mission Planner Integration

### Pre-Flight Checklist
1. ✅ Pixhawk 2.4.8 connected to Pi via TELEM2
2. ✅ Pi Camera working (`python main.py --test-camera`)
3. ✅ ML model loaded on Pi (`model/` directory)
4. ✅ Waypoint mission planned in Mission Planner
5. ✅ Telemetry radio connected for ground station
6. ✅ Battery fully charged

### Flying the Mission
1. Plan waypoints in Mission Planner (10-15m altitude recommended)
2. Upload mission to Pixhawk
3. SSH into Pi and start detection: `python main.py`
4. Arm and switch to AUTO mode from Mission Planner
5. Monitor detection log on Pi console or via telemetry

### After Landing
Check the detection log:
```bash
cat detection_log.csv
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `kagglehub` login prompt | Place `kaggle.json` in `~/.kaggle/` |
| CUDA out of memory | Reduce `BATCH_SIZE` to 4 in `config/training_config.py` |
| Pixhawk won't connect | Check UART wiring, ensure serial is enabled on Pi |
| Pi Camera not detected | Run `libcamera-hello` to test, check CSI cable |
| Low FPS on Pi | Use NCNN model format, reduce `CAMERA_RESOLUTION` |
| Too many false positives | Increase `CONFIDENCE_THRESHOLD` or `MIN_DETECTION_COUNT` |
