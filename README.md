# Building Image Classifier

End-to-end deep learning pipeline for classifying buildings from images using transfer learning (EfficientNet, ResNet, MobileNet). Built with PyTorch and designed for practical reproducibility.

## Overview

This project provides a complete image classification pipeline for building types:

- **Transfer learning** with pretrained backbones (EfficientNet-B0 by default)
- **Configurable** via YAML — swap backbones, tune hyperparameters, change data paths
- **Modular** data pipeline supporting any folder-structured dataset
- **Evaluation** with accuracy, precision, recall, F1, and confusion matrix
- **Inference** for single images or entire directories
- **Streamlit demo** for interactive prediction
- **CI/CD** with GitHub Actions

## Recommended Dataset

### Intel Image Classification (Primary Recommendation)

| Property | Value |
|----------|-------|
| **Name** | Intel Image Classification |
| **Source** | [Kaggle — puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |
| **License** | CC BY-SA 4.0 |
| **Classes** | `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` |
| **Size** | ~25,000 images (150x150px) |
| **Why chosen** | Clean labels, balanced classes, well-documented, widely used benchmark |

The project also supports **any custom dataset** in ImageFolder format — see [Dataset Format](#dataset-format).

## Setup

### Prerequisites
- Python 3.9+
- pip

### Install

```bash
git clone https://github.com/vineel31/Classifying-Buildings-from-Images
cd Classifying-Buildings-from-Images
pip install -r requirements.txt
```

## Dataset Setup

### Option A: Intel Image Classification (Kaggle)

```bash
# Install Kaggle CLI and set up API key (~/.kaggle/kaggle.json)
pip install kaggle

# Download dataset
python scripts/download_dataset.py --method kaggle --output data/raw

# Reorganize into train/val/test splits
python scripts/reorganize_intel_dataset.py --input data/raw --output data/dataset
```

### Option B: Custom Building Dataset

Organize your images:
```
data/dataset/
  train/
    residential/
      img001.jpg
    commercial/
      img001.jpg
    industrial/
      ...
  val/
    residential/
    commercial/
    industrial/
  test/
    residential/
    commercial/
    industrial/
```

### Option C: Mock Data (Smoke Testing / Dev)

```bash
make mock-data
# Generates synthetic colored images at data/mock_dataset/
```

## Dataset Format

The project auto-detects two layouts:

**Pre-split** (preferred):
```
data/dataset/train/<class>/ | val/<class>/ | test/<class>/
```

**Flat** (auto-split by config):
```
data/dataset/<class>/
```

## Training

```bash
# Train with default config
python -m src.training.train --config configs/default.yaml

# Override options
python -m src.training.train \
  --config configs/default.yaml \
  --data-dir data/dataset \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 64

# Quick smoke test on mock data
make train-mock
```

Key config options (`configs/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | `efficientnet_b0` | Backbone model |
| `data.image_size` | `224` | Input image size |
| `data.batch_size` | `32` | Training batch size |
| `training.epochs` | `30` | Max training epochs |
| `training.learning_rate` | `0.001` | Initial learning rate |
| `training.early_stopping_patience` | `7` | Epochs without improvement before stopping |

## Evaluation

```bash
python -m src.training.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_model.pth \
  --output-dir outputs/evaluation
```

Produces:
- `outputs/evaluation/evaluation_metrics.json` — accuracy, F1, per-class breakdown
- `outputs/evaluation/confusion_matrix.png` — normalized confusion matrix

## Inference

```bash
# Single image
python -m src.inference.predict --image path/to/building.jpg

# Directory of images
python -m src.inference.predict --dir path/to/images/ --output results.json

# Via Makefile
make infer IMAGE=path/to/building.jpg
```

## Demo App

```bash
streamlit run app.py
```

Open `http://localhost:8501` — upload any building image for instant classification.

## Supported Backbones

| Backbone | Params | Notes |
|----------|--------|-------|
| `efficientnet_b0` | ~5M | Default — best speed/accuracy tradeoff |
| `efficientnet_b2` | ~9M | Higher accuracy, slower |
| `resnet50` | ~25M | Classic, robust |
| `resnet18` | ~11M | Fastest, good baseline |
| `mobilenetv3_large_100` | ~5M | Mobile-optimized |
| `convnext_tiny` | ~28M | Modern CNN, strong accuracy |

## Repository Structure

```
Classifying-Buildings-from-Images/
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset loading, transforms, DataLoaders
│   │   └── mock_data.py        # Synthetic data generator for testing
│   ├── models/
│   │   └── classifier.py       # BuildingClassifier (transfer learning)
│   ├── training/
│   │   ├── train.py            # Training loop, checkpointing, early stopping
│   │   └── evaluate.py         # Evaluation with full metrics
│   ├── inference/
│   │   └── predict.py          # Single-image and batch inference
│   └── utils/
│       ├── config.py           # YAML config loading
│       ├── metrics.py          # Accuracy, precision, recall, F1
│       └── visualization.py    # Training curves, confusion matrix
├── configs/
│   └── default.yaml            # Default training configuration
├── scripts/
│   ├── download_dataset.py     # Dataset download helper
│   └── reorganize_intel_dataset.py  # Intel dataset reorganizer
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training_smoke.py
│   └── test_inference.py
├── .github/workflows/ci.yml    # GitHub Actions CI
├── app.py                      # Streamlit demo
├── Makefile                    # Convenience commands
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=src tests/

# Specific module
pytest tests/test_model.py -v
```

## Known Limitations

1. **No pretrained building-specific weights** — uses ImageNet-pretrained backbones; domain gap may affect accuracy on unusual building types
2. **Intel dataset has 6 classes**, only one is "buildings" — for a buildings-only classifier, use a dedicated dataset or filter classes
3. **Full training not run** in this repo commit — model pipeline is smoke-tested, not trained to convergence
4. **Mock data is synthetic** — results on mock data are not representative of real performance

## Expected Results (Intel Dataset, EfficientNet-B0, 30 epochs)

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 88-93% |
| Buildings class F1 | 85-92% |
| Training time (GPU) | ~20-40 min |
| Training time (CPU) | ~4-8 hours |

*Note: Results from literature — actual results depend on hardware and hyperparameters.*

## Future Improvements

- [ ] Weights & Biases / MLflow experiment tracking
- [ ] Multi-label classification support
- [ ] Test-time augmentation (TTA)
- [ ] ONNX / TorchScript export for deployment
- [ ] Docker image for reproducible environment
- [ ] Dataset class filtering (buildings-only from Intel dataset)
- [ ] Grad-CAM visualization for model explainability

## License

MIT License — see [LICENSE](LICENSE)
