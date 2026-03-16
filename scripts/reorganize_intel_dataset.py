"""Reorganize Intel Image Classification dataset into standard train/val/test structure.

The Intel dataset has: seg_train/seg_train/ and seg_test/seg_test/
We reorganize into: train/ val/ test/ with class subdirectories.
"""
import argparse
import random
import shutil
from pathlib import Path


INTEL_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
BUILDINGS_ONLY_MODE = False  # Set True to extract only building images


def reorganize(input_dir: str, output_dir: str, val_fraction: float = 0.15, seed: int = 42) -> None:
    random.seed(seed)
    src = Path(input_dir)
    out = Path(output_dir)

    train_src = src / "seg_train" / "seg_train"
    test_src = src / "seg_test" / "seg_test"

    if not train_src.exists():
        raise FileNotFoundError(f"Expected seg_train/seg_train/ in {input_dir}")

    out.mkdir(parents=True, exist_ok=True)

    # Process train -> train + val split
    for cls_dir in sorted(train_src.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = list(cls_dir.glob("*.*"))
        random.shuffle(images)
        n_val = int(len(images) * val_fraction)
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = out / split / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

    # Process test
    if test_src.exists():
        for cls_dir in sorted(test_src.iterdir()):
            if not cls_dir.is_dir():
                continue
            dest = out / "test" / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.glob("*.*"):
                shutil.copy2(img, dest / img.name)

    print(f"Dataset reorganized to {out}")
    # Print summary
    for split in ["train", "val", "test"]:
        split_path = out / split
        if split_path.exists():
            total = sum(len(list((split_path / cls).glob("*.*"))) for cls in INTEL_CLASSES if (split_path / cls).exists())
            print(f"  {split}: {total} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to downloaded Intel dataset")
    parser.add_argument("--output", type=str, default="data/intel_buildings", help="Output directory")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    reorganize(args.input, args.output, args.val_fraction)


if __name__ == "__main__":
    main()
