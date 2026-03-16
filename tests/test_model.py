"""Tests for model initialization and forward pass."""
import pytest
import torch

from src.models.classifier import BuildingClassifier, create_model


class TestBuildingClassifier:
    def test_init_default(self):
        model = BuildingClassifier(num_classes=6)
        assert model.num_classes == 6

    def test_forward_pass(self):
        model = BuildingClassifier(num_classes=4, backbone="efficientnet_b0", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 4)

    def test_param_counts(self):
        model = BuildingClassifier(num_classes=3, pretrained=False)
        counts = model.get_num_params()
        assert counts["total"] > 0
        assert counts["trainable"] <= counts["total"]

    def test_frozen_backbone(self):
        model = BuildingClassifier(num_classes=3, pretrained=False, freeze_backbone=True)
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad

    def test_create_model_factory(self):
        device = torch.device("cpu")
        model = create_model(num_classes=5, backbone="efficientnet_b0", pretrained=False, device=device)
        assert isinstance(model, BuildingClassifier)

    @pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet18", "mobilenetv3_large_100"])
    def test_multiple_backbones(self, backbone):
        model = BuildingClassifier(num_classes=3, backbone=backbone, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 3)
