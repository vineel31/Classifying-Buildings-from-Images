"""Building image classifier using transfer learning."""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = [
    "efficientnet_b0",
    "efficientnet_b2",
    "resnet50",
    "resnet18",
    "mobilenetv3_large_100",
    "convnext_tiny",
]


class BuildingClassifier(nn.Module):
    """Transfer learning model for building image classification.

    Uses a pretrained backbone (EfficientNet, ResNet, etc.) with a
    custom classification head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if not HAS_TIMM:
            raise ImportError("timm is required. Install with: pip install timm")

        # Load pretrained backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # Probe actual output feature dim via a dummy forward pass (some backbones
        # report num_features incorrectly for certain timm versions).
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224)
            _out = self.backbone(_dummy)
            feature_dim = _out.shape[-1]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Backbone {backbone} frozen.")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        logger.info(f"Built BuildingClassifier: backbone={backbone}, num_classes={num_classes}, feature_dim={feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_num_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def create_model(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> BuildingClassifier:
    """Factory function to create and optionally load a model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BuildingClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model.to(device)
