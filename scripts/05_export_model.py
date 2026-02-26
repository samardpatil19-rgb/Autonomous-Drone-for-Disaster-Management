"""
Script 05: Export Model for Raspberry Pi 5
============================================
Exports the trained YOLOv8n model to optimized formats
for edge deployment on Raspberry Pi 5.

Formats:
    - NCNN (fastest on ARM — recommended for Pi 5)
    - ONNX (widely compatible backup format)

Usage:
    python scripts/05_export_model.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import (
    get_best_model_path, IMAGE_SIZE, EXPORT_FORMATS,
    PROJECT_ROOT, RUNS_DIR, TRAINING_NAME
)


def export_model():
    """Export trained model to Pi-optimized formats."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  Model Export — Optimized for Raspberry Pi 5")
    print("=" * 60)
    print()

    # Load best model
    model_path = get_best_model_path()
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run 'python scripts/03_train_model.py' first.")
        sys.exit(1)

    print(f"Source model: {model_path}")
    print(f"Model size:   {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    print(f"Export formats: {EXPORT_FORMATS}")
    print()

    model = YOLO(model_path)
    export_dir = os.path.join(PROJECT_ROOT, "exported_models")
    os.makedirs(export_dir, exist_ok=True)

    exported_paths = {}

    for i, fmt in enumerate(EXPORT_FORMATS, 1):
        print(f"[{i}/{len(EXPORT_FORMATS)}] Exporting to {fmt.upper()}...")
        print("-" * 40)

        try:
            export_path = model.export(
                format=fmt,
                imgsz=IMAGE_SIZE,
                half=False,        # Full precision (Pi 5 doesn't benefit much from FP16)
                simplify=True if fmt == "onnx" else False,
            )

            exported_paths[fmt] = export_path
            print(f"  ✅ {fmt.upper()} exported successfully!")

            # Show file size
            if os.path.isfile(export_path):
                size_mb = os.path.getsize(export_path) / (1024 * 1024)
                print(f"  Size: {size_mb:.1f} MB")
            elif os.path.isdir(export_path):
                total_size = sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, _, files in os.walk(export_path)
                    for f in files
                )
                print(f"  Size: {total_size / (1024 * 1024):.1f} MB")

            print()

        except Exception as e:
            print(f"  ❌ Failed to export {fmt}: {e}")
            print(f"  This is OK — you can use another format on the Pi.")
            print()

    # Verify exported model with a test inference
    print("=" * 60)
    print("  Verifying exported model...")
    print("=" * 60)
    print()

    # Try to run inference with the first available exported model
    import numpy as np

    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    verified = False

    for fmt, path in exported_paths.items():
        try:
            print(f"  Testing {fmt.upper()} model...")
            test_model = YOLO(path)
            results = test_model(test_image, verbose=False)
            print(f"  ✅ {fmt.upper()} inference works! (detected {len(results[0].boxes)} objects in test)")
            verified = True
            break
        except Exception as e:
            print(f"  ⚠️  {fmt.upper()} verification failed: {e}")

    print()

    # Summary
    print("=" * 60)
    print("  📦 Export Summary")
    print("=" * 60)
    print()
    for fmt, path in exported_paths.items():
        print(f"  {fmt.upper():>6}: {path}")

    print()
    print("  📋 To deploy on Raspberry Pi 5:")
    print("     1. Copy the exported model folder to the Pi")
    print("     2. Copy the pi_deployment/ folder to the Pi")
    print("     3. Update the model path in pi_deployment/config.py")
    print("     4. Run: python main.py")
    print()
    print("  Next step: Run 'python scripts/06_test_webcam.py' to test locally.")
    print("=" * 60)

    return exported_paths


if __name__ == "__main__":
    export_model()
