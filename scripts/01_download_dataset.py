"""
Script 01: Download SARD Dataset from Kaggle
=============================================
Downloads the SARD (Search and Rescue Dataset) using kagglehub.

Prerequisites:
    - pip install kagglehub
    - Kaggle API credentials configured (or will prompt for login)

Usage:
    python scripts/01_download_dataset.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.training_config import DATASET_DIR


def download_dataset():
    """Download SARD dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("=" * 60)
    print("  SARD Dataset Downloader")
    print("  (Search and Rescue Drone Dataset)")
    print("=" * 60)
    print()

    # Download the dataset
    print("[1/3] Downloading SARD dataset from Kaggle...")
    print("      This may take a few minutes depending on your connection.")
    print()

    path = kagglehub.dataset_download("nikolasgegenava/sard-search-and-rescue")

    print(f"\n[2/3] Dataset downloaded to: {path}")
    print()

    # List contents
    print("[3/3] Dataset contents:")
    print("-" * 40)

    total_files = 0
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        file_count = len(files)
        total_files += file_count

        if level <= 2:  # Only show top 2 levels
            print(f"{indent}📁 {folder_name}/ ({file_count} files)")

            # Show sample files (first 3)
            if level <= 1:
                for f in sorted(files)[:3]:
                    print(f"{indent}  📄 {f}")
                if file_count > 3:
                    print(f"{indent}  ... and {file_count - 3} more files")

    print("-" * 40)
    print(f"Total files: {total_files}")
    print()
    print(f"✅ Dataset ready at: {path}")
    print()
    print("Next step: Run 'python scripts/02_prepare_dataset.py' to prepare the dataset.")

    return path


if __name__ == "__main__":
    download_dataset()
