"""
Script 02: Prepare and Verify Dataset
=======================================
Validates the downloaded SARD dataset structure, checks annotations,
and creates/verifies the data.yaml configuration for YOLOv8 training.

Usage:
    python scripts/02_prepare_dataset.py
"""

import os
import sys
import glob
import shutil
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import DATASET_DIR, PROJECT_ROOT


def find_dataset_path():
    """Find the downloaded SARD dataset path."""
    # Check if dataset is already in project directory
    if os.path.exists(DATASET_DIR):
        yaml_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.yaml"), recursive=True)
        if yaml_files:
            return DATASET_DIR

    # Check common kagglehub download locations
    home = os.path.expanduser("~")
    kaggle_paths = [
        os.path.join(home, ".cache", "kagglehub", "datasets", "nikolasgegenava", "sard-search-and-rescue"),
        os.path.join(home, ".kaggle", "datasets", "nikolasgegenava", "sard-search-and-rescue"),
    ]

    for base_path in kaggle_paths:
        if os.path.exists(base_path):
            # Find the latest version
            versions = glob.glob(os.path.join(base_path, "versions", "*"))
            if versions:
                latest = sorted(versions)[-1]
                return latest
            return base_path

    return None


def validate_yolo_annotation(label_file):
    """Validate a single YOLO annotation file."""
    errors = []
    line_count = 0

    try:
        with open(label_file, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Line {i}: Expected 5 values, got {len(parts)}")
                    continue

                try:
                    cls_id = int(parts[0])
                    x, y, w, h = [float(p) for p in parts[1:]]
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        errors.append(f"Line {i}: Values out of [0,1] range")
                except ValueError:
                    errors.append(f"Line {i}: Non-numeric values found")
    except Exception as e:
        errors.append(f"Cannot read file: {e}")

    return line_count, errors


def prepare_dataset():
    """Prepare and validate the SARD dataset for YOLOv8 training."""
    print("=" * 60)
    print("  SARD Dataset Preparation & Validation")
    print("=" * 60)
    print()

    # Step 1: Find dataset
    print("[1/5] Locating dataset...")
    dataset_path = find_dataset_path()

    if dataset_path is None:
        print("ERROR: Dataset not found!")
        print("Please run 'python scripts/01_download_dataset.py' first.")
        sys.exit(1)

    print(f"  Found at: {dataset_path}")
    print()

    # Step 2: Explore structure
    print("[2/5] Exploring dataset structure...")
    print("-" * 40)

    # Find all image and label directories
    image_dirs = []
    label_dirs = []

    for root, dirs, files in os.walk(dataset_path):
        rel = os.path.relpath(root, dataset_path)
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in files)
        has_labels = any(f.lower().endswith('.txt') for f in files)

        if has_images:
            image_dirs.append(root)
            print(f"  📁 Images: {rel} ({sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))} files)")
        if has_labels:
            label_dirs.append(root)
            print(f"  📁 Labels: {rel} ({sum(1 for f in files if f.lower().endswith('.txt'))} files)")

    print()

    # Step 3: Check for existing data.yaml
    print("[3/5] Looking for data.yaml configuration...")
    yaml_files = glob.glob(os.path.join(dataset_path, "**", "*.yaml"), recursive=True)
    yaml_files += glob.glob(os.path.join(dataset_path, "**", "*.yml"), recursive=True)

    data_yaml_path = None
    if yaml_files:
        data_yaml_path = yaml_files[0]
        print(f"  Found: {data_yaml_path}")
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"  Contents: {data_config}")
    else:
        print("  No data.yaml found — will create one.")

    print()

    # Step 4: Validate annotations
    print("[4/5] Validating YOLO annotations...")

    total_annotations = 0
    total_objects = 0
    error_files = 0
    class_counts = {}
    sample_checked = 0

    for label_dir in label_dirs:
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        for lf in label_files:
            if os.path.basename(lf) in ("classes.txt", "notes.txt"):
                continue

            sample_checked += 1
            obj_count, errors = validate_yolo_annotation(lf)
            total_annotations += 1
            total_objects += obj_count

            if errors:
                error_files += 1
                if error_files <= 3:
                    print(f"  ⚠️  {os.path.basename(lf)}: {errors[0]}")

            # Count classes
            try:
                with open(lf, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls = parts[0]
                            class_counts[cls] = class_counts.get(cls, 0) + 1
            except Exception:
                pass

    print(f"\n  Annotations checked: {total_annotations}")
    print(f"  Total objects: {total_objects}")
    print(f"  Files with errors: {error_files}")
    print(f"  Class distribution: {class_counts}")
    print()

    # Step 5: Create/update data.yaml for training
    print("[5/5] Creating training data.yaml...")

    # Determine the dataset structure for data.yaml
    train_images = None
    val_images = None
    test_images = None

    for img_dir in image_dirs:
        dir_lower = img_dir.lower()
        if "train" in dir_lower:
            train_images = img_dir
        elif "val" in dir_lower or "valid" in dir_lower:
            val_images = img_dir
        elif "test" in dir_lower:
            test_images = img_dir

    # If no split found, use all images as train
    if train_images is None and image_dirs:
        train_images = image_dirs[0]
        print("  ⚠️  No train/val split found. Using all images for training.")

    # Create data.yaml in project directory
    output_yaml = os.path.join(PROJECT_ROOT, "data.yaml")

    data_yaml_content = {
        "path": dataset_path,
        "train": os.path.relpath(train_images, dataset_path) if train_images else "images/train",
        "val": os.path.relpath(val_images, dataset_path) if val_images else os.path.relpath(train_images, dataset_path) if train_images else "images/val",
        "nc": 1,
        "names": {0: "person"},
    }

    if test_images:
        data_yaml_content["test"] = os.path.relpath(test_images, dataset_path)

    with open(output_yaml, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved to: {output_yaml}")
    print(f"  Config:")
    for k, v in data_yaml_content.items():
        print(f"    {k}: {v}")

    print()
    print("=" * 60)
    print("  ✅ Dataset preparation complete!")
    print(f"  Training config: {output_yaml}")
    print()
    print("  Next step: Run 'python scripts/03_train_model.py' to start training.")
    print("=" * 60)

    return output_yaml


if __name__ == "__main__":
    prepare_dataset()
