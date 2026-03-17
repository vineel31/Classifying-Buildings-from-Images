"""Unit tests for run_pipeline.py — JSONL loading, label counting, resume logic."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from run_pipeline import _count_labels, _reload_all_results, load_already_done


class TestLoadAlreadyDone(unittest.TestCase):
    """Verify resume logic reads previously completed images."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        self.tmppath = Path(self.tmpfile.name)

    def tearDown(self):
        self.tmpfile.close()
        os.unlink(self.tmppath)

    def test_empty_file(self):
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertEqual(result, set())

    def test_nonexistent_file(self):
        result = load_already_done(Path("/nonexistent/file.jsonl"))
        self.assertEqual(result, set())

    def test_reads_successful_records(self):
        record = {"location_id": "loc_001", "image_file": "2022.jpg", "error": None}
        self.tmpfile.write(json.dumps(record) + "\n")
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertIn(("loc_001", "2022.jpg"), result)

    def test_skips_error_records(self):
        record = {"location_id": "loc_001", "image_file": "2022.jpg", "error": "API failed"}
        self.tmpfile.write(json.dumps(record) + "\n")
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertEqual(len(result), 0)

    def test_multiple_records(self):
        for i in range(3):
            record = {"location_id": f"loc_{i:03d}", "image_file": f"{i}.jpg", "error": None}
            self.tmpfile.write(json.dumps(record) + "\n")
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertEqual(len(result), 3)

    def test_ignores_blank_lines(self):
        record = {"location_id": "loc_001", "image_file": "a.jpg", "error": None}
        self.tmpfile.write(json.dumps(record) + "\n\n\n")
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertEqual(len(result), 1)

    def test_ignores_bad_json(self):
        self.tmpfile.write("not-valid-json\n")
        record = {"location_id": "loc_001", "image_file": "a.jpg", "error": None}
        self.tmpfile.write(json.dumps(record) + "\n")
        self.tmpfile.close()
        result = load_already_done(self.tmppath)
        self.assertEqual(len(result), 1)


class TestReloadAllResults(unittest.TestCase):
    """Verify full result reloading from JSONL."""

    def test_groups_by_location(self):
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        records = [
            {"location_id": "loc_001", "image_file": "a.jpg"},
            {"location_id": "loc_001", "image_file": "b.jpg"},
            {"location_id": "loc_002", "image_file": "c.jpg"},
        ]
        for r in records:
            tmpfile.write(json.dumps(r) + "\n")
        tmpfile.close()

        locations = {"loc_001": {}, "loc_002": {}}
        result = _reload_all_results(Path(tmpfile.name), locations)

        self.assertEqual(len(result["loc_001"]), 2)
        self.assertEqual(len(result["loc_002"]), 1)
        os.unlink(tmpfile.name)

    def test_ignores_unknown_locations(self):
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        record = {"location_id": "loc_999", "image_file": "x.jpg"}
        tmpfile.write(json.dumps(record) + "\n")
        tmpfile.close()

        locations = {"loc_001": {}}
        result = _reload_all_results(Path(tmpfile.name), locations)

        self.assertEqual(len(result["loc_001"]), 0)
        self.assertNotIn("loc_999", result)
        os.unlink(tmpfile.name)


class TestCountLabels(unittest.TestCase):
    """Verify label frequency counting."""

    def test_counts_correct(self):
        results = [
            {"final_label": "single_family"},
            {"final_label": "single_family"},
            {"final_label": "commercial"},
        ]
        counts = _count_labels(results)
        self.assertEqual(counts["single_family"], 2)
        self.assertEqual(counts["commercial"], 1)

    def test_empty_list(self):
        self.assertEqual(_count_labels([]), {})

    def test_all_same_label(self):
        results = [{"final_label": "unknown"}] * 5
        counts = _count_labels(results)
        self.assertEqual(counts, {"unknown": 5})


if __name__ == "__main__":
    unittest.main()
