"""Training script for building image classifier."""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import build_dataloaders, build_dataloaders_from_splits
from src.models.classifier import create_model
from src.utils.config import load_config
from src.utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
) -> Dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = {"loss": total_loss / total, "accuracy": correct / total}
    metrics.update(compute_metrics(all_labels, all_preds))
    return metrics


def train(config: dict) -> None:
    """Main training function."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    set_seed(train_cfg["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    # Build data loaders - support both flat and pre-split layouts
    data_dir = data_cfg["data_dir"]
    has_splits = (Path(data_dir) / "train").exists()

    if has_splits:
        train_loader, val_loader, test_loader, class_to_idx = build_dataloaders_from_splits(
            data_dir=data_dir,
            image_size=data_cfg["image_size"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            aug_config=data_cfg.get("augmentation"),
        )
    else:
        train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
            data_dir=data_dir,
            image_size=data_cfg["image_size"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            train_split=data_cfg["train_split"],
            val_split=data_cfg["val_split"],
            seed=train_cfg["seed"],
            aug_config=data_cfg.get("augmentation"),
        )

    num_classes = model_cfg.get("num_classes") or len(class_to_idx)
    logger.info(f"Classes ({num_classes}): {list(class_to_idx.keys())}")

    # Save class mapping
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # Build model
    model = create_model(
        num_classes=num_classes,
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg.get("dropout", 0.3),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        device=device,
    )
    param_counts = model.get_num_params()
    logger.info(f"Model params: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 1e-4)
    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    # Scheduler
    epochs = train_cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 7)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics.get('f1', 0):.4f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "class_to_idx": class_to_idx,
                    "config": config,
                },
                ckpt_dir / "best_model.pth",
            )
            logger.info(f"  Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Save training history
    log_dir = Path(train_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    if test_loader:
        logger.info("Running final test evaluation...")
        ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test results: {test_metrics}")
        with open(log_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Plot training curves
    try:
        from src.utils.visualization import plot_training_curves
        plot_training_curves(history, save_dir=str(log_dir))
    except Exception as e:
        logger.warning(f"Could not plot training curves: {e}")

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train building image classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--backbone", type=str, help="Override backbone model")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--checkpoint-dir", type=str, help="Override checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir

    train(config)


if __name__ == "__main__":
    main()
