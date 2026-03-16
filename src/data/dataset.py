"""Dataset loading and preprocessing for building image classification."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)


def get_transforms(image_size: int = 224, augment: bool = True, aug_config: Optional[dict] = None) -> transforms.Compose:
    """Build image transformation pipeline."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augment and aug_config:
        cj = aug_config.get("color_jitter", {})
        train_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip() if aug_config.get("random_horizontal_flip", True) else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug_config.get("random_rotation", 15)),
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
            ),
            transforms.ToTensor(),
            normalize,
        ]
        return transforms.Compose(train_transforms)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


class BuildingDataset(Dataset):
    """Dataset for building image classification using folder structure.

    Expected folder structure:
        data_dir/
            class_a/
                image1.jpg
                image2.png
            class_b/
                image3.jpg
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_dataset(class_to_idx)

    def _load_dataset(self, class_to_idx: Optional[Dict[str, int]] = None) -> None:
        """Scan directory and collect (image_path, label) pairs."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"No class subdirectories found in {self.data_dir}")

        self.classes = [d.name for d in class_dirs]
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_dir in class_dirs:
            if class_dir.name not in self.class_to_idx:
                logger.warning(f"Skipping unknown class directory: {class_dir.name}")
                continue
            label = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.data_dir}")

        logger.info(f"Loaded {len(self.samples)} images across {len(self.classes)} classes from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}") from e

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            counts[idx_to_class[label]] += 1
        return counts


def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    aug_config: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Build train/val/test DataLoaders from a flat dataset directory."""
    full_dataset = BuildingDataset(
        data_dir=data_dir,
        transform=get_transforms(image_size, augment=False),
    )
    class_to_idx = full_dataset.class_to_idx

    total = len(full_dataset)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    # Apply augmentation to training set by replacing the transform
    train_transform = get_transforms(image_size, augment=True, aug_config=aug_config)
    train_ds.dataset = BuildingDataset(data_dir=data_dir, transform=train_transform, class_to_idx=class_to_idx)

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")
    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False), class_to_idx


def build_dataloaders_from_splits(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    aug_config: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Build DataLoaders from pre-split train/val/test subdirectories."""
    data_path = Path(data_dir)

    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"

    if not train_path.exists():
        raise FileNotFoundError(f"Expected train/ subdirectory in {data_dir}")

    train_transform = get_transforms(image_size, augment=True, aug_config=aug_config)
    val_transform = get_transforms(image_size, augment=False)

    train_ds = BuildingDataset(str(train_path), transform=train_transform)
    class_to_idx = train_ds.class_to_idx

    val_ds = BuildingDataset(str(val_path), transform=val_transform, class_to_idx=class_to_idx) if val_path.exists() else None
    test_ds = BuildingDataset(str(test_path), transform=val_transform, class_to_idx=class_to_idx) if test_path.exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_ds else None

    return train_loader, val_loader, test_loader, class_to_idx
