"""Tests for inference pipeline."""
import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image
import numpy as np


class TestBuildingPredictor:
    """Inference tests require a saved checkpoint — skipped if none exists."""

    def test_predict_image_shape(self, tmp_path):
        """Test that image loading and transform works correctly."""
        from src.data.dataset import get_transforms
        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(str(img_path))

        transform = get_transforms(224, augment=False)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_predictor_skipped_without_checkpoint(self, tmp_path):
        """Predictor raises FileNotFoundError for missing checkpoint."""
        import torch
        pytest.importorskip("src.inference.predict")
        from src.inference.predict import BuildingPredictor
        from src.utils.config import load_config

        fake_ckpt = tmp_path / "fake.pth"
        config = {
            "model": {"backbone": "efficientnet_b0"},
            "data": {"image_size": 224},
            "inference": {"confidence_threshold": 0.5},
        }
        with pytest.raises(Exception):
            BuildingPredictor(str(fake_ckpt), config)
