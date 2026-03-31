"""
Script 03: Train YOLOv8s on SARD Dataset (v2 — High Accuracy)
===============================================================
Fine-tunes the pre-trained YOLOv8s model on the SARD dataset
for person detection in aerial/drone imagery.

Features:
    - Auto-resume: If training was interrupted, rerun this script
      and it will automatically resume from the last checkpoint.
    - Saves best.pt and last.pt after every epoch.

Hardware: RTX 3050 (4GB VRAM) — uses batch_size=2 with imgsz=1280

Usage:
    python scripts/03_train_model.py             # Train or auto-resume
    python scripts/03_train_model.py --resume     # Force resume from checkpoint
    python scripts/03_train_model.py --fresh      # Force fresh training
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import (
    MODEL_VARIANT, EPOCHS, BATCH_SIZE, IMAGE_SIZE,
    DEVICE, WORKERS, PATIENCE, TRAINING_NAME, PROJECT_ROOT, RUNS_DIR,
    FREEZE_LAYERS, get_last_model_path
)


def train_model(force_resume=False, force_fresh=False):
    """Fine-tune YOLOv8s on SARD dataset with resume support."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  YOLOv8s Training — Disaster Rescue Person Detector (v2)")
    print("=" * 60)
    print()

    # Check for data.yaml
    data_yaml = os.path.join(PROJECT_ROOT, "data.yaml")
    if not os.path.exists(data_yaml):
        print("ERROR: data.yaml not found!")
        print("Please run 'python scripts/02_prepare_dataset.py' first.")
        sys.exit(1)

    # Check if we can resume from a previous training
    last_model = get_last_model_path()
    can_resume = os.path.exists(last_model) and not force_fresh

    if can_resume:
        print("*" * 60)
        print("  RESUME MODE: Found previous checkpoint!")
        print(f"  Checkpoint: {last_model}")
        print("  Training will continue from where it left off.")
        print("  (Use --fresh flag to start over)")
        print("*" * 60)
        print()

    # Print training configuration
    print("Training Configuration:")
    print(f"  Model:       {MODEL_VARIANT}")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print(f"  Image Size:  {IMAGE_SIZE}")
    print(f"  Device:      GPU {DEVICE}")
    print(f"  Workers:     {WORKERS}")
    print(f"  Patience:    {PATIENCE} (early stopping)")
    print(f"  Freeze:      {FREEZE_LAYERS} backbone layers (retains COCO knowledge)")
    print(f"  Output:      {RUNS_DIR}")
    print(f"  Mode:        {'RESUME' if can_resume else 'FRESH START'}")
    print()

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  WARNING: No GPU detected — training will be SLOW on CPU!")
            print("  Consider using Google Colab for faster training.")
    except ImportError:
        pass

    print()
    print("-" * 60)

    if can_resume and (force_resume or True):
        # RESUME from last checkpoint
        print("Resuming training from last checkpoint...")
        print("-" * 60)
        print()

        model = YOLO(last_model)
        results = model.train(resume=True)

    else:
        # FRESH training
        print("Starting fresh training... (estimated ~8-10 hours)")
        print("-" * 60)
        print()

        model = YOLO(MODEL_VARIANT)
        results = model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            patience=PATIENCE,
            name=TRAINING_NAME,
            project=os.path.join(RUNS_DIR, "detect"),
            exist_ok=True,
            # Augmentation settings for aerial imagery
            flipud=0.5,        # Vertical flip (common in aerial views)
            fliplr=0.5,        # Horizontal flip
            mosaic=1.0,         # Mosaic augmentation
            scale=0.5,          # Scale augmentation
            hsv_h=0.015,        # HSV-Hue augmentation
            hsv_s=0.7,          # HSV-Saturation
            hsv_v=0.4,          # HSV-Value (brightness)
            # Freeze backbone layers (retains COCO person detection for webcam)
            freeze=FREEZE_LAYERS,
            # Save settings
            save=True,
            save_period=10,     # Save checkpoint every 10 epochs as backup
            plots=True,         # Generate training plots
            verbose=True,
        )

    print()
    print("=" * 60)
    print("  Training Complete!")
    weights_dir = os.path.join(RUNS_DIR, 'detect', TRAINING_NAME, 'weights')
    print(f"  Best weights: {os.path.join(weights_dir, 'best.pt')}")
    print(f"  Last weights: {os.path.join(weights_dir, 'last.pt')}")
    print(f"  Training plots: {os.path.join(RUNS_DIR, 'detect', TRAINING_NAME)}")
    print()
    print("  To evaluate: python scripts/04_evaluate_model.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8s person detector")
    parser.add_argument("--resume", action="store_true",
                        help="Force resume from last checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Force fresh training (ignore checkpoints)")
    args = parser.parse_args()

    train_model(force_resume=args.resume, force_fresh=args.fresh)
