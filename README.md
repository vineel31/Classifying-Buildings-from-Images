# Building Classification from Street View Images

AI pipeline for automatically classifying building types (residential, commercial, mixed-use) from Google Street View imagery using temporal analysis and vision language models.

## Overview

This system analyzes multi-year Street View images (2007-2024) to classify building use types with high confidence. It combines:
- **Temporal aggregation** - weighs recent images higher
- **Vision language models** - Llama 4 Scout via Groq API
- **OCR analysis** - detects business signage
- **Year-based reasoning** - captures changes over time

**Dataset:** 10 locations, 88 images across 18 years  
**Result Distribution:** 6 single-family, 4 mixed-use  
**Average Confidence:** 0.85 accuracy

---

## Quick Start

### Prerequisites
- Python 3.9+
- Groq API key (free at https://console.groq.com)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Run Pipeline

```bash
# Main pipeline (all images with OCR boost)
python run_pipeline.py --data ./data --output ./results

# Variation test (single most recent image, no OCR)
python run_pipeline.py --data ./data --output ./results_variation --single-image

# Resume after interruption
python run_pipeline.py --data ./data --output ./results --resume

Output files: `predictions.csv`, `intermediate_results.jsonl`, `summary.json`

---

## Project Structure

```
├── src/
│   ├── config.py              # Configuration & API keys
│   ├── data_loader.py         # Stage 1: Load & organize images
│   ├── model_analyzer.py      # Stage 2: Vision analysis (Groq/Llama)
│   └── classifier.py          # Stage 3+4: Temporal aggregation & classification
├── run_pipeline.py            # Main entry point
├── data/                      # 10 location folders with images
├── results/                   # Main pipeline outputs
├── results_variation/         # Variation test outputs
└── requirements.txt
```

---

## Pipeline Design

**Stage 1: Data Loader**
- Reads metadata.csv and discovers image files
- Groups images by location and extracts year from filename

**Stage 2: Vision Analyzer**
- Encodes each image as base64
- Sends to Llama 4 Scout with structured JSON prompt
- Extracts: OCR text, visual cues, building features, per-image classification

**Stage 3: Temporal Aggregator**
- Weights confidence by year (2018+: 1.0x, 2013-2018: 0.75x, pre-2013: 0.5x)
- Detects temporal changes across years
- Collects OCR text pool from all years

**Stage 4: Classifier**
- Applies OCR boost rule (business names → mixed_use)
- Enforces confidence threshold (< 0.60 → unknown)
- Outputs final label with reasoning and confidence

---

## Classification Labels

| Label | Description |
|-------|-------------|
| `single_family` | Detached residential home |
| `apartment_condo` | Multi-unit residential building |
| `commercial` | Office, retail, or industrial |
| `mixed_use` | Residential + commercial combined |
| `empty_land` | Vacant lot or under construction |
| `unknown` | Low confidence or unclear |

---

## Key Features

- **Multi-year temporal analysis** - All available images (2007-2024) analyzed
- **Smart year weighting** - Recent evidence prioritized
- **OCR business detection** - Signage recognition for commercial classification
- **Crash-safe resume** - JSONL format enables interrupted runs
- **Controlled variation** - Single-image mode for comparison studies
- **Detailed reasoning** - Every prediction includes explanatory text

---

## Results

### Main Pipeline (88 images)
- **Total locations:** 10
- **Distribution:** 6 single-family, 4 mixed-use
- **Confidence range:** 0.65-0.93
- **Processing time:** ~7 minutes
- **Success rate:** 100% (0 errors)

### Variation Test (10 images)
- **Configuration:** Single most recent image per location, no OCR boost
- **Result:** 10 single-family, 0 mixed-use
- **Finding:** 4 locations reclassified (temporal + OCR effect validated)

---

## API Information

- **Model:** Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct)
- **Provider:** Groq API
- **Rate limit:** 30 req/min (pipeline uses 5s delay = 12 RPM)
- **Cost:** ~$0.001/image on v1/vision endpoint

---

## Output Files

**predictions.csv**
- location_id, address, final_label, final_confidence, images_analyzed, reasoning

**intermediate_results.jsonl**
- Per-image analysis: OCR text, detected objects, visual cues, building features, per-image label

**summary.json**
- Metadata: mode, total_locations, total_images, label distribution, all results

---

## Requirements

```
groq>=0.4.0
Pillow>=10.0.0
python-dotenv>=1.0.0
reportlab>=4.0.0
```

---

## License

MIT
