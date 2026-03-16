"""Stages 3+4: Temporal aggregation and final classification."""

from collections import Counter
from typing import List, Dict
from src.config import VALID_LABELS, CONFIDENCE_THRESHOLD, get_year_weight


def aggregate_and_classify(location_id: str, image_analyses: List[dict], disable_ocr_boost: bool = False) -> dict:
    """Aggregate per-image analyses into a single final classification for a location."""
    if not image_analyses:
        return _make_result(location_id, "unknown", 0.0, "No images available.", {})

    valid_analyses = [a for a in image_analyses if a.get("analysis")]
    if not valid_analyses:
        return _make_result(location_id, "unknown", 0.0, "All image analyses failed.", {})

    label_scores: Dict[str, float] = Counter()
    total_weight = 0.0
    all_ocr_text = []
    all_objects  = []
    all_cues     = {"residential_indicators": [], "commercial_indicators": []}
    years_seen   = []
    label_by_year = {}

    for item in valid_analyses:
        year     = item.get("year", 0)
        analysis = item.get("analysis", {})
        label    = analysis.get("preliminary_classification", "unknown")
        conf     = float(analysis.get("confidence", 0.5))
        weight   = get_year_weight(year)

        if label in VALID_LABELS:
            label_scores[label] += conf * weight
        total_weight += weight
        years_seen.append(year)
        label_by_year[year] = label

        ocr = analysis.get("ocr_text", {})
        if ocr.get("detected"):
            all_ocr_text.extend(ocr.get("text_found", []))
            all_ocr_text.extend(ocr.get("signage", []))
            all_ocr_text.extend(ocr.get("business_names", []))

        all_objects.extend(analysis.get("detected_objects", []))
        cues = analysis.get("visual_cues", {})
        all_cues["residential_indicators"].extend(cues.get("residential_indicators", []))
        all_cues["commercial_indicators"].extend(cues.get("commercial_indicators", []))

    if total_weight > 0:
        for label in label_scores:
            label_scores[label] /= total_weight

    best_label = max(label_scores, key=label_scores.get) if label_scores else "unknown"
    best_score = label_scores.get(best_label, 0.0)

    unique_labels  = list(dict.fromkeys(label_by_year[y] for y in sorted(label_by_year)))
    temporal_change = len(set(unique_labels)) > 1

    temporal_summary = {
        "years_analyzed":           sorted(years_seen),
        "label_by_year":            {str(y): l for y, l in sorted(label_by_year.items())},
        "temporal_change_detected": temporal_change,
        "unique_labels_seen":       unique_labels,
        "label_scores":             {k: round(v, 3) for k, v in label_scores.items()},
        "ocr_text_found":           list(set(filter(None, all_ocr_text))),
        "common_objects":           _top_items(all_objects, n=10),
        "top_residential_cues":     _top_items(all_cues["residential_indicators"], n=5),
        "top_commercial_cues":      _top_items(all_cues["commercial_indicators"], n=5),
    }

    ocr_unique = temporal_summary["ocr_text_found"]
    if ocr_unique and best_label == "single_family" and not disable_ocr_boost:
        best_label = "mixed_use"
        best_score = min(best_score * 0.85, 0.75)

    if best_score < CONFIDENCE_THRESHOLD:
        reasoning = (
            f"Low confidence ({best_score:.2f}) across {len(valid_analyses)} images. "
            f"Scores: {temporal_summary['label_scores']}. Defaulting to unknown."
        )
        return _make_result(location_id, "unknown", best_score, reasoning, temporal_summary)

    change_note = ""
    if temporal_change:
        change_note = f" Building changed over time: {unique_labels}. Using recent evidence."

    reasoning = (
        f"Classified as '{best_label}' with confidence {best_score:.2f} "
        f"from {len(valid_analyses)} image(s) across years {sorted(years_seen)}.{change_note}"
    )
    return _make_result(location_id, best_label, best_score, reasoning, temporal_summary)


def _make_result(location_id, label, confidence, reasoning, temporal_summary):
    return {
        "location_id":      location_id,
        "final_label":      label,
        "final_confidence": round(confidence, 3),
        "reasoning":        reasoning,
        "temporal_summary": temporal_summary,
    }


def _top_items(items: list, n: int = 5) -> list:
    """Return top-n most frequent items."""
    if not items:
        return []
    return [item for item, _ in Counter(items).most_common(n)]
