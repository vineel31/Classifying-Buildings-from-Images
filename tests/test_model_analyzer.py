"""Unit tests for src/model_analyzer.py — response parsing, fallback, retry delay."""

import json
import unittest

from src.model_analyzer import (
    _fallback_analysis,
    _parse_response,
    _parse_retry_delay,
)


class TestParseResponse(unittest.TestCase):
    """Verify JSON parsing of model responses."""

    VALID_RESPONSE = json.dumps({
        "ocr_text": {"detected": False, "text_found": [], "signage": [], "business_names": []},
        "detected_objects": ["house", "tree"],
        "visual_cues": {
            "num_entrances": 1, "has_storefront": False, "has_porch": True,
            "has_garage": True, "has_balcony": False, "parking_type": "driveway",
            "signage_visible": False, "residential_indicators": ["porch"],
            "commercial_indicators": [],
        },
        "building_features": {
            "building_present": True, "num_stories": 1,
            "architectural_style": "ranch", "primary_material": "vinyl_siding",
            "building_condition": "good",
        },
        "preliminary_classification": "single_family",
        "confidence": 0.92,
        "reasoning": "Detached house with porch and driveway.",
    })

    def test_valid_json_parsed(self):
        result = _parse_response(self.VALID_RESPONSE, "test.jpg")
        self.assertEqual(result["preliminary_classification"], "single_family")
        self.assertAlmostEqual(result["confidence"], 0.92)

    def test_markdown_wrapped_json(self):
        wrapped = f"```json\n{self.VALID_RESPONSE}\n```"
        result = _parse_response(wrapped, "test.jpg")
        self.assertEqual(result["preliminary_classification"], "single_family")

    def test_invalid_json_returns_fallback(self):
        result = _parse_response("this is not json", "test.jpg")
        self.assertEqual(result["preliminary_classification"], "unknown")
        self.assertEqual(result["confidence"], 0.0)

    def test_empty_response_returns_fallback(self):
        result = _parse_response("", "test.jpg")
        self.assertEqual(result["preliminary_classification"], "unknown")

    def test_none_response_returns_fallback(self):
        result = _parse_response(None, "test.jpg")
        self.assertEqual(result["preliminary_classification"], "unknown")

    def test_invalid_label_normalized_to_unknown(self):
        data = json.loads(self.VALID_RESPONSE)
        data["preliminary_classification"] = "skyscraper"
        result = _parse_response(json.dumps(data), "test.jpg")
        self.assertEqual(result["preliminary_classification"], "unknown")

    def test_confidence_clamped_high(self):
        data = json.loads(self.VALID_RESPONSE)
        data["confidence"] = 5.0
        result = _parse_response(json.dumps(data), "test.jpg")
        self.assertLessEqual(result["confidence"], 1.0)

    def test_confidence_clamped_low(self):
        data = json.loads(self.VALID_RESPONSE)
        data["confidence"] = -0.5
        result = _parse_response(json.dumps(data), "test.jpg")
        self.assertGreaterEqual(result["confidence"], 0.0)


class TestFallbackAnalysis(unittest.TestCase):
    """Verify fallback analysis structure."""

    def test_structure_has_all_keys(self):
        result = _fallback_analysis("unknown", 0.0, "test error")
        for key in ("ocr_text", "detected_objects", "visual_cues",
                     "building_features", "preliminary_classification",
                     "confidence", "reasoning"):
            self.assertIn(key, result)

    def test_fallback_label_preserved(self):
        result = _fallback_analysis("commercial", 0.3, "partial failure")
        self.assertEqual(result["preliminary_classification"], "commercial")

    def test_fallback_confidence_preserved(self):
        result = _fallback_analysis("unknown", 0.5, "timeout")
        self.assertEqual(result["confidence"], 0.5)

    def test_ocr_not_detected(self):
        result = _fallback_analysis("unknown", 0.0, "error")
        self.assertFalse(result["ocr_text"]["detected"])

    def test_no_objects_detected(self):
        result = _fallback_analysis("unknown", 0.0, "error")
        self.assertEqual(result["detected_objects"], [])


class TestParseRetryDelay(unittest.TestCase):
    """Verify retry delay extraction from error messages."""

    def test_extracts_retry_after(self):
        error = "Rate limit exceeded. Retry after: 30 seconds"
        delay = _parse_retry_delay(error)
        self.assertEqual(delay, 32)  # 30 + 2 buffer

    def test_default_when_no_match(self):
        from src.config import RETRY_DELAY
        delay = _parse_retry_delay("Generic error with no timing info")
        self.assertEqual(delay, RETRY_DELAY)

    def test_empty_error_string(self):
        from src.config import RETRY_DELAY
        delay = _parse_retry_delay("")
        self.assertEqual(delay, RETRY_DELAY)


if __name__ == "__main__":
    unittest.main()
