"""
Script 04: Evaluate Trained Model
===================================
Runs validation on the test/validation split and reports
mAP, precision, recall, and F1 metrics.

Usage:
    python scripts/04_evaluate_model.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import (
    get_best_model_path, IMAGE_SIZE, CONFIDENCE_THRESHOLD,
    DEVICE, PROJECT_ROOT, RUNS_DIR, TRAINING_NAME
)


def evaluate_model():
    """Evaluate the trained model on validation data."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  Model Evaluation — Disaster Rescue Person Detector")
    print("=" * 60)
    print()

    # Load best model
    model_path = get_best_model_path()
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python scripts/03_train_model.py' first.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Check for data.yaml
    data_yaml = os.path.join(PROJECT_ROOT, "data.yaml")
    if not os.path.exists(data_yaml):
        print("ERROR: data.yaml not found!")
        sys.exit(1)

    print(f"Dataset config: {data_yaml}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()

    # Run validation
    print("-" * 60)
    print("Running validation...")
    print("-" * 60)
    print()

    results = model.val(
        data=data_yaml,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        plots=True,
        save_json=True,
        name=f"{TRAINING_NAME}_eval",
        project=os.path.join(RUNS_DIR, "detect"),
        exist_ok=True,
    )

    # Print metrics
    print()
    print("=" * 60)
    print("  📊 Evaluation Results")
    print("=" * 60)
    print()

    # Access metrics
    metrics = results.results_dict if hasattr(results, 'results_dict') else {}

    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-' * 42}")

    metric_names = {
        'metrics/precision(B)': 'Precision',
        'metrics/recall(B)': 'Recall',
        'metrics/mAP50(B)': 'mAP@50',
        'metrics/mAP50-95(B)': 'mAP@50-95',
    }

    for key, name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {name:<30} {value:>8.4f}  {bar}")

    print()

    # Model info
    print("  Model Info:")
    print(f"    Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"    Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")

    # Evaluation outputs
    eval_dir = os.path.join(RUNS_DIR, "detect", f"{TRAINING_NAME}_eval")
    print()
    print(f"  📁 Evaluation outputs saved to:")
    print(f"     {eval_dir}")
    print()

    # List generated plots
    if os.path.exists(eval_dir):
        plots = [f for f in os.listdir(eval_dir) if f.endswith('.png')]
        if plots:
            print("  Generated plots:")
            for p in sorted(plots):
                print(f"    📊 {p}")

    print()
    print("  Next step: Run 'python scripts/05_export_model.py' to export for Pi.")
    print("=" * 60)

    return results


if __name__ == "__main__":
    evaluate_model()
