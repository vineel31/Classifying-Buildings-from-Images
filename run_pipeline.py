"""
Building Classification Pipeline

Usage:
    python run_pipeline.py --data ./data --output ./results
    python run_pipeline.py --data ./data --output ./results --resume
    python run_pipeline.py --data ./data --output ./results_variation --single-image
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from src.data_loader import load_dataset
from src.model_analyzer import analyze_image
from src.classifier import aggregate_and_classify


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify building types from Google Street View images."
    )
    parser.add_argument("--data",         required=True, help="Dataset directory (contains metadata.csv)")
    parser.add_argument("--output",       required=True, help="Output directory")
    parser.add_argument("--single-image", action="store_true", help="Process most recent image only, disable OCR boost")
    parser.add_argument("--resume",       action="store_true", help="Skip already-analyzed images")
    return parser.parse_args()


def load_already_done(jsonl_path: Path) -> set:
    """Return set of (location_id, filename) tuples already successfully analyzed."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("error") is None:
                    done.add((record["location_id"], record["image_file"]))
            except json.JSONDecodeError:
                continue
    return done


def main():
    args              = parse_args()
    data_dir          = args.data
    output_dir        = args.output
    single_image_mode = args.single_image
    resume_mode       = args.resume

    os.makedirs(output_dir, exist_ok=True)

    if single_image_mode:
        mode_label = "SINGLE-IMAGE (no OCR boost)"
    elif resume_mode:
        mode_label = "MULTI-IMAGE (resume)"
    else:
        mode_label = "MULTI-IMAGE"

    print(f"\n{'='*60}")
    print(f"  Building Classification Pipeline")
    print(f"  Mode   : {mode_label}")
    print(f"  Data   : {data_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    print("[Stage 1] Loading dataset...")
    try:
        locations = load_dataset(data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[Stage 2] Analyzing images...")

    intermediate_path = Path(output_dir) / "intermediate_results.jsonl"
    already_done = set()

    if resume_mode:
        already_done = load_already_done(intermediate_path)
        print(f"  Resume mode: {len(already_done)} images already completed, skipping them.\n")

    open_mode = "a" if resume_mode else "w"
    all_image_results = {}

    with open(intermediate_path, open_mode, encoding="utf-8") as jsonl_file:
        for loc_id, loc_data in locations.items():
            images = loc_data["images"]
            if single_image_mode and images:
                images = [max(images, key=lambda x: x["year"])]

            if not images:
                print(f"  [{loc_id}] No images found, skipping.")
                all_image_results[loc_id] = []
                continue

            if resume_mode:
                images = [img for img in images if (loc_id, img["filename"]) not in already_done]

            skipped   = len(loc_data["images"]) - len(images)
            skip_note = f" ({skipped} already done)" if skipped else ""
            print(f"  [{loc_id}] {loc_data['address']} — {len(images)} image(s){skip_note}")

            image_analyses = []
            for img_info in images:
                print(f"    Analyzing {img_info['filename']} ({img_info['year']})...")
                result = analyze_image(img_info, loc_id)

                if result.get("error"):
                    print(f"    [Warning] {result['error'][:80]}...")

                image_analyses.append(result)
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            all_image_results[loc_id] = image_analyses

    if resume_mode:
        all_image_results = _reload_all_results(intermediate_path, locations)

    print(f"\n[Stage 3+4] Classifying locations...")

    final_results = []
    for loc_id, loc_data in locations.items():
        image_analyses = all_image_results.get(loc_id, [])
        classification = aggregate_and_classify(
            loc_id, image_analyses, disable_ocr_boost=single_image_mode
        )

        final_results.append({
            "location_id":      loc_id,
            "address":          loc_data["address"],
            "final_label":      classification["final_label"],
            "final_confidence": classification["final_confidence"],
            "images_analyzed":  len(image_analyses),
            "reasoning":        classification["reasoning"],
        })

        label = classification["final_label"]
        conf  = classification["final_confidence"]
        print(f"  [{loc_id}] {loc_data['address']} -> {label} ({conf:.2f})")

    csv_path   = Path(output_dir) / "predictions.csv"
    fieldnames = ["location_id", "address", "final_label", "final_confidence", "images_analyzed", "reasoning"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    mode_str     = "single_image" if single_image_mode else "multi_image"
    summary_path = Path(output_dir) / "summary.json"
    summary = {
        "mode":            mode_str,
        "total_locations": len(final_results),
        "total_images":    sum(r["images_analyzed"] for r in final_results),
        "label_counts":    _count_labels(final_results),
        "results":         final_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Pipeline complete | {len(final_results)} locations | {summary['label_counts']}")
    print(f"Predictions: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}\n")


def _reload_all_results(jsonl_path: Path, locations: dict) -> dict:
    """Reload results from JSONL grouped by location_id."""
    all_results = {loc_id: [] for loc_id in locations}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                loc_id = record.get("location_id")
                if loc_id in all_results:
                    all_results[loc_id].append(record)
            except json.JSONDecodeError:
                continue
    return all_results


def _count_labels(results: list) -> dict:
    counts = {}
    for r in results:
        label = r["final_label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()
