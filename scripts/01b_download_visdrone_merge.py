"""
Script 01b: Download VisDrone Dataset & Merge with SARD
========================================================
Downloads the VisDrone-DET dataset, extracts only person/pedestrian
annotations, and merges them with the existing SARD dataset for
a larger, more diverse training set.

VisDrone classes we need:
    - Class 0: pedestrian (single person walking/standing)
    - Class 1: people (group of people)

These get remapped to our Class 0: person

Usage:
    python scripts/01b_download_visdrone_merge.py
"""

import os
import sys
import glob
import shutil
import yaml
import zipfile
import urllib.request

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import PROJECT_ROOT


# VisDrone-DET download URLs (from official GitHub)
VISDRONE_URLS = {
    "train_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test_images": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}

# VisDrone original classes (we only want pedestrian=0 and people=1)
PERSON_CLASSES = {0, 1}  # pedestrian, people


def download_file(url, dest_path):
    """Download a file with progress display."""
    if os.path.exists(dest_path):
        print(f"  Already downloaded: {os.path.basename(dest_path)}")
        return

    print(f"  Downloading: {os.path.basename(dest_path)}")
    print(f"  URL: {url}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook)
    print()


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting: {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def convert_visdrone_annotation(visdrone_label_file):
    """
    Convert VisDrone annotation format to YOLO format.
    
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
    
    Returns list of YOLO-format lines for person detections only.
    """
    yolo_lines = []

    try:
        with open(visdrone_label_file, 'r') as f:
            lines = f.readlines()
    except Exception:
        return []

    # We need image dimensions — try to infer from corresponding image
    # For now, we'll need to process with image dimensions
    return lines  # Will be processed per-image


def convert_visdrone_to_yolo(annotations_dir, images_dir, output_labels_dir):
    """
    Convert all VisDrone annotations to YOLO format,
    keeping only pedestrian and people classes.
    
    Returns count of converted files.
    """
    from PIL import Image

    os.makedirs(output_labels_dir, exist_ok=True)
    converted = 0
    person_count = 0

    annotation_files = glob.glob(os.path.join(annotations_dir, "*.txt"))

    for ann_file in annotation_files:
        basename = os.path.splitext(os.path.basename(ann_file))[0]

        # Find corresponding image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(images_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # Get image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
            img.close()
        except Exception:
            continue

        # Convert annotations
        yolo_lines = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                score = int(parts[4])
                category = int(parts[5])

                # Skip if score is 0 (ignored region) or not person class
                if score == 0:
                    continue
                if category not in PERSON_CLASSES:
                    continue

                # Convert to YOLO format (normalized)
                x_center = (bbox_left + bbox_width / 2) / img_w
                y_center = (bbox_top + bbox_height / 2) / img_h
                w_norm = bbox_width / img_w
                h_norm = bbox_height / img_h

                # Clamp to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                # Skip tiny/invalid boxes
                if w_norm < 0.005 or h_norm < 0.005:
                    continue

                # Class 0 = person (remapped from both pedestrian and people)
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                person_count += 1

        # Write YOLO annotation file (even if empty — background image)
        output_file = os.path.join(output_labels_dir, basename + ".txt")
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        converted += 1

    return converted, person_count


def merge_datasets(sard_path, visdrone_path, output_path):
    """Merge SARD and VisDrone into a single combined dataset."""
    print("\n[MERGE] Combining SARD + VisDrone datasets...")

    for split in ['train', 'valid', 'test']:
        vd_split = split if split != 'valid' else 'val'

        # Create output directories
        out_images = os.path.join(output_path, split, "images")
        out_labels = os.path.join(output_path, split, "labels")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

        count = 0

        # Copy SARD images and labels
        sard_img_dir = os.path.join(sard_path, "search-and-rescue", split, "images")
        sard_lbl_dir = os.path.join(sard_path, "search-and-rescue", split, "labels")

        if os.path.exists(sard_img_dir):
            for img_file in glob.glob(os.path.join(sard_img_dir, "*.*")):
                ext = os.path.splitext(img_file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    basename = os.path.splitext(os.path.basename(img_file))[0]
                    # Copy with prefix to avoid name collisions
                    new_name = f"sard_{basename}"
                    shutil.copy2(img_file, os.path.join(out_images, new_name + ext))

                    # Copy corresponding label
                    label_file = os.path.join(sard_lbl_dir, basename + ".txt")
                    if os.path.exists(label_file):
                        shutil.copy2(label_file, os.path.join(out_labels, new_name + ".txt"))
                    else:
                        # Create empty label (background image)
                        open(os.path.join(out_labels, new_name + ".txt"), 'w').close()
                    count += 1

        # Copy VisDrone images and labels
        vd_img_dir = os.path.join(visdrone_path, vd_split, "images")
        vd_lbl_dir = os.path.join(visdrone_path, vd_split, "labels")

        if os.path.exists(vd_img_dir):
            for img_file in glob.glob(os.path.join(vd_img_dir, "*.*")):
                ext = os.path.splitext(img_file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    basename = os.path.splitext(os.path.basename(img_file))[0]
                    new_name = f"vd_{basename}"
                    shutil.copy2(img_file, os.path.join(out_images, new_name + ext))

                    label_file = os.path.join(vd_lbl_dir, basename + ".txt")
                    if os.path.exists(label_file):
                        shutil.copy2(label_file, os.path.join(out_labels, new_name + ".txt"))
                    else:
                        open(os.path.join(out_labels, new_name + ".txt"), 'w').close()
                    count += 1

        print(f"  {split}: {count} images")

    # Create combined data.yaml
    combined_yaml = {
        "path": output_path,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "person"},
    }

    yaml_path = os.path.join(PROJECT_ROOT, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(combined_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Combined data.yaml saved to: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("  VisDrone Dataset Download & Merge with SARD")
    print("=" * 60)
    print()

    visdrone_dir = os.path.join(PROJECT_ROOT, "datasets", "visdrone")
    visdrone_raw = os.path.join(visdrone_dir, "raw")
    visdrone_yolo = os.path.join(visdrone_dir, "yolo")
    combined_dir = os.path.join(PROJECT_ROOT, "datasets", "combined")
    os.makedirs(visdrone_raw, exist_ok=True)

    # Step 1: Download VisDrone
    print("[1/4] Downloading VisDrone-DET dataset...")
    print("      (This is ~2 GB total, may take 10-20 minutes)")
    print()

    for name, url in VISDRONE_URLS.items():
        zip_name = url.split("/")[-1]
        zip_path = os.path.join(visdrone_raw, zip_name)
        download_file(url, zip_path)

        # Extract
        if not os.path.exists(os.path.join(visdrone_raw, zip_name.replace('.zip', ''))):
            extract_zip(zip_path, visdrone_raw)

    print()

    # Step 2: Convert VisDrone annotations to YOLO format
    print("[2/4] Converting VisDrone annotations to YOLO format...")
    print("      (Extracting only pedestrian/people classes)")
    print()

    splits_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev",
    }

    total_converted = 0
    total_persons = 0

    for split_name, folder_name in splits_map.items():
        raw_folder = os.path.join(visdrone_raw, folder_name)
        ann_dir = os.path.join(raw_folder, "annotations")
        img_dir = os.path.join(raw_folder, "images")

        if not os.path.exists(ann_dir):
            print(f"  WARNING: {ann_dir} not found, skipping {split_name}")
            continue

        out_img_dir = os.path.join(visdrone_yolo, split_name, "images")
        out_lbl_dir = os.path.join(visdrone_yolo, split_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)

        # Copy images
        print(f"  Processing {split_name}...")
        for img in glob.glob(os.path.join(img_dir, "*.*")):
            shutil.copy2(img, out_img_dir)

        # Convert annotations
        conv, persons = convert_visdrone_to_yolo(ann_dir, img_dir, out_lbl_dir)
        total_converted += conv
        total_persons += persons
        print(f"    Converted: {conv} files, {persons} person annotations")

    print(f"\n  Total: {total_converted} images, {total_persons} person annotations")
    print()

    # Step 3: Find SARD dataset
    print("[3/4] Locating SARD dataset...")

    home = os.path.expanduser("~")
    sard_path = os.path.join(home, ".cache", "kagglehub", "datasets",
                              "nikolasgegenava", "sard-search-and-rescue", "versions", "1")

    if not os.path.exists(sard_path):
        print(f"  ERROR: SARD not found at {sard_path}")
        print("  Please run 'python scripts/01_download_dataset.py' first")
        sys.exit(1)

    print(f"  Found SARD at: {sard_path}")
    print()

    # Step 4: Merge datasets
    print("[4/4] Merging SARD + VisDrone into combined dataset...")
    yaml_path = merge_datasets(sard_path, visdrone_yolo, combined_dir)

    print()
    print("=" * 60)
    print("  Dataset merge complete!")
    print(f"  Combined dataset: {combined_dir}")
    print(f"  Training config:  {yaml_path}")
    print()
    print("  Next: Run 'python scripts/03_train_model.py' to train")
    print("=" * 60)


if __name__ == "__main__":
    main()
