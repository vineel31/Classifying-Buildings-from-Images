import os
import csv
from pathlib import Path
from typing import Dict, List


def load_dataset(data_dir: str) -> Dict[str, dict]:
    """Return dict keyed by location_id with address, folder, and image list."""
    data_dir = Path(data_dir).resolve()
    metadata_path = data_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    locations = {}

    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc_id  = row.get("location_id", "").strip()
            address = row.get("address", "").strip()
            folder  = row.get("folder_name", "").strip()

            if not loc_id or not folder:
                continue

            folder_path = data_dir / folder
            locations[loc_id] = {
                "location_id": loc_id,
                "address":     address,
                "folder":      folder,
                "folder_path": str(folder_path),
                "images":      _load_images_from_folder(folder_path),
            }

    if not locations:
        raise ValueError(f"No valid locations found in {metadata_path}")

    print(f"[DataLoader] Loaded {len(locations)} locations, "
          f"{sum(len(v['images']) for v in locations.values())} images total.")
    return locations


def _load_images_from_folder(folder_path: Path) -> List[dict]:
    """Return list of images sorted chronologically by year."""
    images = []

    if not folder_path.exists():
        print(f"  [Warning] Folder not found: {folder_path}")
        return images

    for img_file in sorted(folder_path.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        images.append({
            "path":     str(img_file),
            "filename": img_file.name,
            "year":     _extract_year(img_file.name),
        })

    images.sort(key=lambda x: x["year"])
    return images


def _extract_year(filename: str) -> int:
    """Extract year from YYYY-MM_<id>.jpg format, or 0 if failed."""
    try:
        return int(filename.split("-")[0])
    except (IndexError, ValueError):
        return 0
