.PHONY: install setup-data mock-data train evaluate infer test lint clean help

help:
	@echo "Building Image Classifier - Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make mock-data     Generate mock dataset for testing"
	@echo "  make train         Train the model (requires dataset)"
	@echo "  make train-mock    Train on mock data (smoke test)"
	@echo "  make evaluate      Evaluate trained model"
	@echo "  make infer IMAGE=path/to/image.jpg   Run inference"
	@echo "  make test          Run test suite"
	@echo "  make lint          Run linter"
	@echo "  make demo          Launch Streamlit demo"
	@echo "  make clean         Remove generated artifacts"

install:
	pip install -r requirements.txt

mock-data:
	python -c "from src.data.mock_data import generate_mock_dataset; generate_mock_dataset('data/mock_dataset')"
	@echo "Mock dataset generated at data/mock_dataset/"

train:
	python -m src.training.train --config configs/default.yaml

train-mock: mock-data
	python -m src.training.train --config configs/default.yaml --data-dir data/mock_dataset --epochs 3

evaluate:
	python -m src.training.evaluate --config configs/default.yaml

infer:
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make infer IMAGE=path/to/image.jpg"; exit 1; fi
	python -m src.inference.predict --image $(IMAGE)

test:
	pytest tests/ -v --tb=short

lint:
	python -m flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503 || true

demo:
	streamlit run app.py

clean:
	rm -rf outputs/ data/mock_dataset/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
