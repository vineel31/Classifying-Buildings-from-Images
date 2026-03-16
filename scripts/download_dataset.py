"""Helper script for downloading recommended datasets.

Recommended dataset: Intel Image Classification (Kaggle)
- 6 classes: buildings, forest, glacier, mountain, sea, street
- ~25K images, 150x150px
- License: CC BY-SA 4.0 (Kaggle dataset)
- Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Alternative: Buildings-Dataset (GitHub / academic)
- Pure buildings focus if needed

Usage:
    python scripts/download_dataset.py --method kaggle --output data/dataset
    python scripts/download_dataset.py --method manual --output data/dataset
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def download_via_kaggle(output_dir: str) -> None:
    """Download Intel Image Classification dataset via Kaggle CLI."""
    try:
        import kaggle
    except ImportError:
        print("kaggle package not found. Install with: pip install kaggle")
        print("Also set up ~/.kaggle/kaggle.json with your API key.")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading Intel Image Classification dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "puneet6060/intel-image-classification", "-p", str(out), "--unzip"],
        check=True,
    )
    print(f"Dataset downloaded to {out}")
    print("\nExpected structure after download:")
    print("  data/dataset/")
    print("    seg_train/seg_train/  (train images)")
    print("    seg_test/seg_test/    (test images)")
    print("    seg_pred/seg_pred/    (unlabeled prediction images)")
    print("\nRun the reorganize script to restructure:")
    print("  python scripts/reorganize_intel_dataset.py --input data/dataset --output data/intel_buildings")


def show_manual_instructions() -> None:
    print("""
Manual Dataset Download Instructions
=====================================

Option 1: Intel Image Classification (Kaggle)
  1. Go to: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
  2. Download and extract to data/dataset/
  3. Run: python scripts/reorganize_intel_dataset.py

Option 2: Use your own building dataset
  Organize images in this folder structure:
    data/dataset/
      train/
        class_name_1/
          image1.jpg
          image2.jpg
        class_name_2/
          ...
      val/
        class_name_1/
        class_name_2/
      test/
        class_name_1/
        class_name_2/

Option 3: Use mock data for testing
  python -c "from src.data.mock_data import generate_mock_dataset; generate_mock_dataset('data/mock_dataset')"
  python -m src.training.train --config configs/default.yaml --data-dir data/mock_dataset
""")


def main():
    parser = argparse.ArgumentParser(description="Download dataset for building classification")
    parser.add_argument("--method", choices=["kaggle", "manual"], default="manual")
    parser.add_argument("--output", type=str, default="data/dataset")
    args = parser.parse_args()

    if args.method == "kaggle":
        download_via_kaggle(args.output)
    else:
        show_manual_instructions()


if __name__ == "__main__":
    main()
