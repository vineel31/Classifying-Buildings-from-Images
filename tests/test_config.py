"""Unit tests for src/config.py — constants and year weight logic."""

import unittest

from src.config import (
    CONFIDENCE_THRESHOLD,
    MAX_RETRIES,
    REQUEST_DELAY,
    RETRY_DELAY,
    VALID_LABELS,
    YEAR_WEIGHT,
    get_year_weight,
)


class TestValidLabels(unittest.TestCase):
    """Verify the set of valid classification labels."""

    EXPECTED_LABELS = [
        "single_family",
        "apartment_condo",
        "commercial",
        "mixed_use",
        "empty_land",
        "unknown",
    ]

    def test_all_expected_labels_present(self):
        for label in self.EXPECTED_LABELS:
            self.assertIn(label, VALID_LABELS)

    def test_no_extra_labels(self):
        self.assertEqual(sorted(VALID_LABELS), sorted(self.EXPECTED_LABELS))

    def test_labels_are_lowercase_snake_case(self):
        for label in VALID_LABELS:
            self.assertEqual(label, label.lower())
            self.assertNotIn(" ", label)


class TestConfidenceThreshold(unittest.TestCase):
    """Verify confidence threshold is in a sensible range."""

    def test_threshold_value(self):
        self.assertEqual(CONFIDENCE_THRESHOLD, 0.60)

    def test_threshold_is_between_0_and_1(self):
        self.assertGreater(CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(CONFIDENCE_THRESHOLD, 1.0)


class TestRetryConstants(unittest.TestCase):
    """Verify retry / delay configuration."""

    def test_max_retries_positive(self):
        self.assertGreater(MAX_RETRIES, 0)

    def test_retry_delay_positive(self):
        self.assertGreater(RETRY_DELAY, 0)

    def test_request_delay_positive(self):
        self.assertGreater(REQUEST_DELAY, 0)


class TestGetYearWeight(unittest.TestCase):
    """Verify year-based weighting function."""

    def test_old_year_low_weight(self):
        self.assertEqual(get_year_weight(2007), 0.5)

    def test_mid_year_medium_weight(self):
        self.assertEqual(get_year_weight(2015), 0.75)

    def test_recent_year_full_weight(self):
        self.assertEqual(get_year_weight(2022), 1.0)

    def test_boundary_2012_is_old(self):
        self.assertEqual(get_year_weight(2012), 0.5)

    def test_boundary_2013_is_mid(self):
        self.assertEqual(get_year_weight(2013), 0.75)

    def test_boundary_2017_is_mid(self):
        self.assertEqual(get_year_weight(2017), 0.75)

    def test_boundary_2018_is_recent(self):
        self.assertEqual(get_year_weight(2018), 1.0)

    def test_out_of_range_defaults_to_1(self):
        self.assertEqual(get_year_weight(1990), 1.0)

    def test_far_future_defaults_to_1(self):
        self.assertEqual(get_year_weight(2050), 1.0)

    def test_year_weight_dict_has_three_ranges(self):
        self.assertEqual(len(YEAR_WEIGHT), 3)


if __name__ == "__main__":
    unittest.main()
