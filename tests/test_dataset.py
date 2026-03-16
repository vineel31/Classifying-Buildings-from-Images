"""Tests for dataset loading and preprocessing."""
import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.mock_data import generate_mock_dataset, MOCK_CLASSES
from src.data.dataset import BuildingDataset, get_transforms, build_dataloaders_from_splits


@pytest.fixture(scope="module")
def mock_dataset_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    generate_mock_dataset(str(tmp), classes=["class_a", "class_b", "class_c"], images_per_class=12, image_size=32)
    return str(tmp)


class TestGetTransforms:
    def test_returns_compose(self):
        t = get_transforms(image_size=224, augment=False)
        from torchvision import transforms
        assert isinstance(t, transforms.Compose)

    def test_output_shape(self):
        from PIL import Image
        t = get_transforms(image_size=64, augment=False)
        img = Image.new("RGB", (100, 100))
        tensor = t(img)
        assert tensor.shape == (3, 64, 64)


class TestBuildingDataset:
    def test_loads_mock_data(self, mock_dataset_dir):
        ds = BuildingDataset(
            str(Path(mock_dataset_dir) / "train"),
            transform=get_transforms(32, augment=False),
        )
        assert len(ds) > 0
        assert len(ds.classes) == 3

    def test_returns_correct_types(self, mock_dataset_dir):
        ds = BuildingDataset(
            str(Path(mock_dataset_dir) / "train"),
            transform=get_transforms(32, augment=False),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)
        assert img.shape[0] == 3  # RGB

    def test_class_distribution(self, mock_dataset_dir):
        ds = BuildingDataset(
            str(Path(mock_dataset_dir) / "train"),
            transform=get_transforms(32, augment=False),
        )
        dist = ds.get_class_distribution()
        assert set(dist.keys()) == {"class_a", "class_b", "class_c"}
        assert all(v > 0 for v in dist.values())

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BuildingDataset(str(tmp_path / "nonexistent"))

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            BuildingDataset(str(tmp_path))


class TestDataLoaders:
    def test_build_dataloaders_from_splits(self, mock_dataset_dir):
        train_loader, val_loader, test_loader, class_to_idx = build_dataloaders_from_splits(
            data_dir=mock_dataset_dir,
            image_size=32,
            batch_size=4,
            num_workers=0,
        )
        assert train_loader is not None
        assert len(class_to_idx) == 3
        batch = next(iter(train_loader))
        images, labels = batch
        assert images.shape[1:] == (3, 32, 32)
