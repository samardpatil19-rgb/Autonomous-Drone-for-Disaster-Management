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
MODEL_VARIANT = "yolov8n.pt"       # YOLOv8 Nano - best for Pi 5
TARGET_CLASSES = [0]                # Class 0 = person/human
CONFIDENCE_THRESHOLD = 0.5         # Minimum confidence for detection

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
EPOCHS = 50                        # Number of training epochs
BATCH_SIZE = 4                     # Batch size (safe for RTX 3050 4GB)
IMAGE_SIZE = 640                   # Input image resolution
DEVICE = 0                         # GPU device (0 = first GPU)
WORKERS = 2                        # DataLoader workers
PATIENCE = 10                      # Early stopping patience

# Training name (used for saving runs)
TRAINING_NAME = "disaster_rescue_detector"

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
def get_best_model_path():
    """Returns the path to the best trained model weights."""
    return os.path.join(RUNS_DIR, "detect", TRAINING_NAME, "weights", "best.pt")

def get_last_model_path():
    """Returns the path to the last trained model weights."""
    return os.path.join(RUNS_DIR, "detect", TRAINING_NAME, "weights", "last.pt")
