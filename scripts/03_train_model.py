"""
Script 03: Train YOLOv8n on SARD Dataset
==========================================
Fine-tunes the pre-trained YOLOv8n model on the SARD dataset
for person detection in aerial/drone imagery.

Hardware: RTX 3050 (4GB VRAM) — uses batch_size=8

Usage:
    python scripts/03_train_model.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import (
    MODEL_VARIANT, EPOCHS, BATCH_SIZE, IMAGE_SIZE,
    DEVICE, WORKERS, PATIENCE, TRAINING_NAME, PROJECT_ROOT, RUNS_DIR
)


def train_model():
    """Fine-tune YOLOv8n on SARD dataset."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  YOLOv8n Training — Disaster Rescue Person Detector")
    print("=" * 60)
    print()

    # Check for data.yaml
    data_yaml = os.path.join(PROJECT_ROOT, "data.yaml")
    if not os.path.exists(data_yaml):
        print("ERROR: data.yaml not found!")
        print("Please run 'python scripts/02_prepare_dataset.py' first.")
        sys.exit(1)

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
    print(f"  Output:      {RUNS_DIR}")
    print()

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  No GPU detected — training will be SLOW on CPU!")
            print("     Consider using Google Colab for faster training.")
    except ImportError:
        pass

    print()
    print("-" * 60)
    print("Starting training... (this will take ~1-2 hours)")
    print("-" * 60)
    print()

    # Load pre-trained YOLOv8n
    model = YOLO(MODEL_VARIANT)

    # Train the model
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
        # Save settings
        save=True,
        save_period=-1,     # Save only best and last
        plots=True,         # Generate training plots
        verbose=True,
    )

    print()
    print("=" * 60)
    print("  ✅ Training Complete!")
    print(f"  Best weights: {os.path.join(RUNS_DIR, 'detect', TRAINING_NAME, 'weights', 'best.pt')}")
    print(f"  Last weights: {os.path.join(RUNS_DIR, 'detect', TRAINING_NAME, 'weights', 'last.pt')}")
    print(f"  Training plots: {os.path.join(RUNS_DIR, 'detect', TRAINING_NAME)}")
    print()
    print("  Next step: Run 'python scripts/04_evaluate_model.py' to evaluate.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_model()
