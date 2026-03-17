"""Unit tests for src/data_loader.py — dataset loading and image discovery."""

import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.data_loader import _extract_year, _load_images_from_folder, load_dataset


class TestExtractYear(unittest.TestCase):
    """Verify year extraction from Street View filenames."""

    def test_standard_filename(self):
        self.assertEqual(_extract_year("2022-10__zzh9-nBI3xOwp_wIMWCRA.jpg"), 2022)

    def test_old_year(self):
        self.assertEqual(_extract_year("2007-06__abc123.jpg"), 2007)

    def test_no_dash_returns_zero(self):
        self.assertEqual(_extract_year("photo.jpg"), 0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(_extract_year(""), 0)

    def test_non_numeric_prefix_returns_zero(self):
        self.assertEqual(_extract_year("abc-10__test.jpg"), 0)


class TestLoadImagesFromFolder(unittest.TestCase):
    """Verify image discovery from a folder."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for f in Path(self.tmpdir).iterdir():
            f.unlink()
        os.rmdir(self.tmpdir)

    def test_finds_jpg_files(self):
        Path(self.tmpdir, "2022-10__abc.jpg").touch()
        Path(self.tmpdir, "2020-05__def.jpeg").touch()
        images = _load_images_from_folder(Path(self.tmpdir))
        self.assertEqual(len(images), 2)

    def test_ignores_non_image_files(self):
        Path(self.tmpdir, "notes.txt").touch()
        Path(self.tmpdir, "data.csv").touch()
        Path(self.tmpdir, "2022-10__abc.jpg").touch()
        images = _load_images_from_folder(Path(self.tmpdir))
        self.assertEqual(len(images), 1)

    def test_sorted_by_year(self):
        Path(self.tmpdir, "2022-10__late.jpg").touch()
        Path(self.tmpdir, "2007-06__early.jpg").touch()
        Path(self.tmpdir, "2015-03__mid.jpg").touch()
        images = _load_images_from_folder(Path(self.tmpdir))
        years = [img["year"] for img in images]
        self.assertEqual(years, [2007, 2015, 2022])

    def test_empty_folder_returns_empty(self):
        images = _load_images_from_folder(Path(self.tmpdir))
        self.assertEqual(images, [])

    def test_nonexistent_folder_returns_empty(self):
        images = _load_images_from_folder(Path("/nonexistent/path"))
        self.assertEqual(images, [])

    def test_image_dict_has_required_keys(self):
        Path(self.tmpdir, "2022-10__abc.jpg").touch()
        images = _load_images_from_folder(Path(self.tmpdir))
        self.assertIn("path", images[0])
        self.assertIn("filename", images[0])
        self.assertIn("year", images[0])

    def test_png_files_included(self):
        Path(self.tmpdir, "2022-10__abc.png").touch()
        images = _load_images_from_folder(Path(self.tmpdir))
        self.assertEqual(len(images), 1)


class TestLoadDataset(unittest.TestCase):
    """Verify full dataset loading from metadata.csv."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_folder = Path(self.tmpdir) / "test_location"
        self.img_folder.mkdir()
        Path(self.img_folder, "2022-10__abc.jpg").touch()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _write_metadata(self, rows):
        path = Path(self.tmpdir) / "metadata.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["location_id", "address", "folder_name"])
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_single_location(self):
        self._write_metadata([{
            "location_id": "loc_001",
            "address": "123 Main St",
            "folder_name": "test_location",
        }])
        result = load_dataset(self.tmpdir)
        self.assertIn("loc_001", result)
        self.assertEqual(result["loc_001"]["address"], "123 Main St")

    def test_images_populated(self):
        self._write_metadata([{
            "location_id": "loc_001",
            "address": "123 Main St",
            "folder_name": "test_location",
        }])
        result = load_dataset(self.tmpdir)
        self.assertEqual(len(result["loc_001"]["images"]), 1)

    def test_missing_metadata_raises(self):
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            load_dataset(empty_dir)
        os.rmdir(empty_dir)

    def test_empty_metadata_raises(self):
        self._write_metadata([])
        with self.assertRaises(ValueError):
            load_dataset(self.tmpdir)

    def test_skips_rows_without_location_id(self):
        self._write_metadata([
            {"location_id": "", "address": "Nowhere", "folder_name": "test_location"},
            {"location_id": "loc_002", "address": "456 Oak Ave", "folder_name": "test_location"},
        ])
        result = load_dataset(self.tmpdir)
        self.assertEqual(len(result), 1)
        self.assertIn("loc_002", result)

    def test_skips_rows_without_folder_name(self):
        self._write_metadata([
            {"location_id": "loc_001", "address": "Nowhere", "folder_name": ""},
        ])
        with self.assertRaises(ValueError):
            load_dataset(self.tmpdir)


if __name__ == "__main__":
    unittest.main()
