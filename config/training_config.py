"""
Training Configuration for Drone Final Year Project
=====================================================
Centralized configuration for model training, evaluation, and export.
Modify values here instead of editing individual scripts.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# ============================================================
# MODEL
# ============================================================
MODEL_VARIANT = "yolov8s.pt"       # YOLOv8 Small - better accuracy
TARGET_CLASSES = [0]                # Class 0 = person/human
CONFIDENCE_THRESHOLD = 0.5         # Minimum confidence for detection

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
EPOCHS = 100                       # Number of training epochs
BATCH_SIZE = 2                     # Batch size (for imgsz 1280 on RTX 3050 4GB)
IMAGE_SIZE = 1280                  # Input image resolution (higher = better accuracy)
DEVICE = 0                         # GPU device (0 = first GPU)
WORKERS = 2                        # DataLoader workers
PATIENCE = 20                      # Early stopping patience (more room for improvement)
FREEZE_LAYERS = 10                 # Freeze first N backbone layers (retains COCO knowledge for webcam)

# Training name (used for saving runs)
TRAINING_NAME = "disaster_rescue_detector_v2"

# ============================================================
# EXPORT (for Raspberry Pi 5)
# ============================================================
EXPORT_FORMATS = ["ncnn", "onnx"]  # Export both NCNN (fastest) and ONNX (backup)

# ============================================================
# WEBCAM TEST
# ============================================================
WEBCAM_INDEX = 0                   # Default webcam
WEBCAM_DISPLAY_WIDTH = 1280
WEBCAM_DISPLAY_HEIGHT = 720

# ============================================================
# HELPER: Get best model path
# ============================================================
def get_best_model_path(version="v2"):
    """Returns the path to the best trained model weights."""
    name = "disaster_rescue_detector" if version == "v1" else TRAINING_NAME
    return os.path.join(RUNS_DIR, "detect", name, "weights", "best.pt")

def get_last_model_path(version="v2"):
    """Returns the path to the last trained model weights."""
    name = "disaster_rescue_detector" if version == "v1" else TRAINING_NAME
    return os.path.join(RUNS_DIR, "detect", name, "weights", "last.pt")
