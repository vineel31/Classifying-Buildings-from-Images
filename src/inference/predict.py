"""Inference script: classify a single image or a directory of images."""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import get_transforms
from src.models.classifier import create_model
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BuildingPredictor:
    """High-level interface for building image classification inference."""

    def __init__(self, checkpoint_path: str, config: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.class_to_idx: Dict[str, int] = ckpt.get("class_to_idx", {})
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        self.class_names: List[str] = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        self.model = create_model(
            num_classes=len(self.class_to_idx),
            backbone=config["model"]["backbone"],
            pretrained=False,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.model.eval()

        self.transform = get_transforms(config["data"]["image_size"], augment=False)
        self.confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.5)

    @torch.no_grad()
    def predict_image(self, image_path: str) -> Dict:
        """Classify a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = probs.max(dim=0)
        predicted_class = self.idx_to_class[top_idx.item()]
        confidence = top_prob.item()

        all_probs = {self.idx_to_class[i]: probs[i].item() for i in range(len(self.class_names))}
        return {
            "image": str(image_path),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "is_confident": confidence >= self.confidence_threshold,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Classify a list of images."""
        return [self.predict_image(p) for p in image_paths]

    def predict_directory(self, directory: str) -> List[Dict]:
        """Classify all images in a directory."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in valid_exts]
        logger.info(f"Found {len(paths)} images in {directory}")
        return self.predict_batch(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on building images")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide --image or --dir")

    config = load_config(args.config)
    checkpoint = args.checkpoint or config["inference"]["checkpoint"]

    predictor = BuildingPredictor(checkpoint, config)

    if args.image:
        result = predictor.predict_image(args.image)
        print(f"\nPrediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
        print("All probabilities:")
        for cls, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {cls}: {prob:.4f}")
        results = [result]
    else:
        results = predictor.predict_directory(args.dir)
        for r in results:
            print(f"{Path(r['image']).name}: {r['predicted_class']} ({r['confidence']:.3f})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
