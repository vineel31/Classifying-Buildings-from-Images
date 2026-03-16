"""Generate a tiny mock dataset for smoke testing."""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


MOCK_CLASSES = ["residential", "commercial", "industrial", "historical", "religious", "skyscraper"]


def generate_mock_dataset(
    output_dir: str,
    classes: list = None,
    images_per_class: int = 20,
    image_size: int = 64,
    seed: int = 42,
) -> str:
    """Generate a tiny synthetic dataset for smoke testing.

    Each class gets solid-color + noise images (not real buildings,
    only for pipeline validation).
    """
    random.seed(seed)
    np.random.seed(seed)

    if classes is None:
        classes = MOCK_CLASSES

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use train/val/test splits
    splits = {"train": int(images_per_class * 0.7), "val": int(images_per_class * 0.15), "test": images_per_class - int(images_per_class * 0.7) - int(images_per_class * 0.15)}

    base_colors = [
        (180, 120, 80),   # residential - brick-ish
        (100, 140, 180),  # commercial - glass-ish
        (80, 80, 80),     # industrial - grey
        (200, 180, 140),  # historical - sandstone
        (220, 200, 160),  # religious - light stone
        (60, 100, 160),   # skyscraper - steel blue
    ]

    for split_name, count in splits.items():
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = out / split_name / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            base_color = base_colors[cls_idx % len(base_colors)]
            for i in range(count):
                noise = np.random.randint(-40, 40, (image_size, image_size, 3), dtype=np.int16)
                img_array = np.clip(np.array(base_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode="RGB")
                img.save(cls_dir / f"img_{i:04d}.png")

    print(f"Mock dataset generated at: {out}")
    print(f"Classes: {classes}")
    print(f"Splits: {splits}")
    return str(out)


if __name__ == "__main__":
    generate_mock_dataset("data/mock_dataset")
