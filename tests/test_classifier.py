"""Unit tests for src/classifier.py — aggregation, OCR boost, and temporal logic."""

import unittest

from src.classifier import _top_items, aggregate_and_classify


def _make_analysis(label, confidence, year, ocr_texts=None, commercial_cues=None,
                   residential_cues=None, objects=None):
    """Helper to build a fake image analysis dict."""
    return {
        "year": year,
        "analysis": {
            "preliminary_classification": label,
            "confidence": confidence,
            "ocr_text": {
                "detected": bool(ocr_texts),
                "text_found": ocr_texts or [],
                "signage": [],
                "business_names": [],
            },
            "detected_objects": objects or [],
            "visual_cues": {
                "residential_indicators": residential_cues or [],
                "commercial_indicators": commercial_cues or [],
            },
        },
    }


class TestAggregateEmpty(unittest.TestCase):
    """Classify with no data."""

    def test_no_analyses_returns_unknown(self):
        result = aggregate_and_classify("loc_001", [])
        self.assertEqual(result["final_label"], "unknown")
        self.assertEqual(result["final_confidence"], 0.0)

    def test_all_failed_analyses_returns_unknown(self):
        analyses = [{"year": 2022, "analysis": None}]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "unknown")


class TestAggregateBasic(unittest.TestCase):
    """Basic single-label aggregation."""

    def test_single_image_classification(self):
        analyses = [_make_analysis("single_family", 0.9, 2022)]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "single_family")
        self.assertGreater(result["final_confidence"], 0.6)

    def test_commercial_classification(self):
        analyses = [_make_analysis("commercial", 0.85, 2020)]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "commercial")

    def test_location_id_preserved(self):
        analyses = [_make_analysis("commercial", 0.85, 2020)]
        result = aggregate_and_classify("loc_007", analyses)
        self.assertEqual(result["location_id"], "loc_007")


class TestAggregateMultipleImages(unittest.TestCase):
    """Aggregation across multiple images with year weighting."""

    def test_majority_label_wins(self):
        analyses = [
            _make_analysis("single_family", 0.8, 2020),
            _make_analysis("single_family", 0.9, 2022),
            _make_analysis("commercial", 0.7, 2010),
        ]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "single_family")

    def test_recent_years_weighted_higher(self):
        analyses = [
            _make_analysis("commercial", 0.9, 2007),    # weight=0.5 → score=0.45
            _make_analysis("single_family", 0.9, 2022), # weight=1.0 → score=0.90
        ]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "single_family")


class TestOCRBoost(unittest.TestCase):
    """OCR boost rule: single_family → mixed_use when OCR text is detected."""

    def test_ocr_boost_triggers(self):
        analyses = [_make_analysis("single_family", 0.9, 2022, ocr_texts=["Open 9-5"])]
        result = aggregate_and_classify("loc_001", analyses, disable_ocr_boost=False)
        self.assertEqual(result["final_label"], "mixed_use")

    def test_ocr_boost_disabled(self):
        analyses = [_make_analysis("single_family", 0.9, 2022, ocr_texts=["Open 9-5"])]
        result = aggregate_and_classify("loc_001", analyses, disable_ocr_boost=True)
        self.assertEqual(result["final_label"], "single_family")

    def test_ocr_boost_only_for_single_family(self):
        analyses = [_make_analysis("commercial", 0.9, 2022, ocr_texts=["Open 9-5"])]
        result = aggregate_and_classify("loc_001", analyses, disable_ocr_boost=False)
        self.assertEqual(result["final_label"], "commercial")

    def test_ocr_boost_caps_confidence(self):
        analyses = [_make_analysis("single_family", 0.95, 2022, ocr_texts=["Shop"])]
        result = aggregate_and_classify("loc_001", analyses, disable_ocr_boost=False)
        self.assertLessEqual(result["final_confidence"], 0.75)


class TestTemporalChange(unittest.TestCase):
    """Temporal change detection."""

    def test_change_detected_in_reasoning(self):
        analyses = [
            _make_analysis("single_family", 0.8, 2007),
            _make_analysis("commercial", 0.9, 2022),
        ]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertIn("changed over time", result["reasoning"])

    def test_no_change_when_labels_same(self):
        analyses = [
            _make_analysis("single_family", 0.8, 2007),
            _make_analysis("single_family", 0.9, 2022),
        ]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertNotIn("changed over time", result["reasoning"])


class TestLowConfidence(unittest.TestCase):
    """Low confidence → unknown fallback."""

    def test_very_low_confidence_returns_unknown(self):
        analyses = [_make_analysis("single_family", 0.3, 2022)]
        result = aggregate_and_classify("loc_001", analyses)
        self.assertEqual(result["final_label"], "unknown")


class TestTopItems(unittest.TestCase):
    """Helper function _top_items."""

    def test_returns_most_frequent(self):
        items = ["car", "car", "tree", "tree", "tree", "fence"]
        result = _top_items(items, n=2)
        self.assertEqual(result[0], "tree")
        self.assertEqual(result[1], "car")

    def test_empty_list(self):
        self.assertEqual(_top_items([]), [])

    def test_n_larger_than_items(self):
        result = _top_items(["a", "b"], n=10)
        self.assertEqual(len(result), 2)


class TestResultStructure(unittest.TestCase):
    """Verify output dict has all required keys."""

    def test_result_keys(self):
        analyses = [_make_analysis("single_family", 0.9, 2022)]
        result = aggregate_and_classify("loc_001", analyses)
        for key in ("location_id", "final_label", "final_confidence", "reasoning", "temporal_summary"):
            self.assertIn(key, result)

    def test_temporal_summary_keys(self):
        analyses = [_make_analysis("single_family", 0.9, 2022)]
        result = aggregate_and_classify("loc_001", analyses)
        ts = result["temporal_summary"]
        for key in ("years_analyzed", "label_by_year", "temporal_change_detected",
                     "label_scores", "ocr_text_found"):
            self.assertIn(key, ts)


if __name__ == "__main__":
    unittest.main()
