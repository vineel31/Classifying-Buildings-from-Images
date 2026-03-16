"""Evaluation script: runs metrics on a checkpoint against a dataset."""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import BuildingDataset, get_transforms
from src.models.classifier import create_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.visualization import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list,
    output_dir: str,
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += images.size(0)

    metrics = {"loss": total_loss / total}
    metrics.update(compute_metrics(all_labels, all_preds, class_names=class_names))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=str(out_dir / "confusion_matrix.png"))
    print_metrics(metrics)
    logger.info(f"Results saved to {out_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate building image classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, help="Path to evaluation dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint or config["evaluation"]["checkpoint"]
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", {})
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    data_dir = args.data_dir or config["data"]["data_dir"]
    # Try test split first
    test_path = Path(data_dir) / "test"
    eval_path = str(test_path) if test_path.exists() else data_dir

    transform = get_transforms(config["data"]["image_size"], augment=False)
    dataset = BuildingDataset(eval_path, transform=transform, class_to_idx=class_to_idx)
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])

    model = create_model(
        num_classes=len(class_to_idx),
        backbone=config["model"]["backbone"],
        pretrained=False,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    run_evaluation(model, loader, device, class_names, args.output_dir or config["evaluation"]["output_dir"])


if __name__ == "__main__":
    main()
