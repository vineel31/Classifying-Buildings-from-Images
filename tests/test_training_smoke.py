"""Smoke tests for the training and inference pipelines."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.mock_data import generate_mock_dataset
from src.data.dataset import build_dataloaders_from_splits
from src.models.classifier import create_model
from src.training.train import train_one_epoch, evaluate, set_seed
from src.utils.config import load_config
from src.utils.metrics import compute_metrics


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    data_dir = tmp / "data"
    ckpt_dir = tmp / "checkpoints"
    log_dir = tmp / "logs"
    ckpt_dir.mkdir()
    log_dir.mkdir()

    generate_mock_dataset(str(data_dir), classes=["class_a", "class_b"], images_per_class=10, image_size=32)

    return {
        "data": {
            "data_dir": str(data_dir),
            "image_size": 32,
            "batch_size": 4,
            "num_workers": 0,
            "train_split": 0.7,
            "val_split": 0.15,
            "augmentation": None,
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "num_classes": 2,
            "dropout": 0.0,
            "freeze_backbone": False,
        },
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping_patience": 2,
            "seed": 42,
            "checkpoint_dir": str(ckpt_dir),
            "log_dir": str(log_dir),
        },
        "evaluation": {"checkpoint": str(ckpt_dir / "best_model.pth"), "output_dir": str(tmp / "eval")},
        "inference": {"checkpoint": str(ckpt_dir / "best_model.pth"), "confidence_threshold": 0.5},
        "_tmp_dir": str(tmp),
    }


class TestTrainingSmoke:
    def test_one_epoch_runs(self, smoke_config):
        set_seed(42)
        device = torch.device("cpu")
        train_loader, val_loader, test_loader, class_to_idx = build_dataloaders_from_splits(
            data_dir=smoke_config["data"]["data_dir"],
            image_size=32,
            batch_size=4,
            num_workers=0,
        )
        model = create_model(num_classes=2, backbone="efficientnet_b0", pretrained=False, device=device)
        import torch.nn as nn
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_runs(self, smoke_config):
        device = torch.device("cpu")
        train_loader, val_loader, test_loader, class_to_idx = build_dataloaders_from_splits(
            data_dir=smoke_config["data"]["data_dir"],
            image_size=32,
            batch_size=4,
            num_workers=0,
        )
        model = create_model(num_classes=2, backbone="efficientnet_b0", pretrained=False, device=device)
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(model, val_loader, criterion, device)
        assert "loss" in metrics
        assert "accuracy" in metrics


class TestMetrics:
    def test_compute_metrics_basic(self):
        labels = [0, 1, 0, 1, 0]
        preds = [0, 1, 1, 1, 0]
        metrics = compute_metrics(labels, preds, class_names=["a", "b"])
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_perfect_predictions(self):
        labels = [0, 1, 2]
        preds = [0, 1, 2]
        metrics = compute_metrics(labels, preds)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
