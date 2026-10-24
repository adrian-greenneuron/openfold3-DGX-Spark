"""Verification must reject missing, failed and different predictions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "compare_predictions", Path(__file__).resolve().parents[1] / "tools/compare_predictions.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PredictionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a, self.b = [Path(self.temp.name) / name for name in ("a", "b")]
        for root in (self.a, self.b):
            root.mkdir()
            (root / "summary.txt").write_text(
                "Total Queries Processed: 1\nSuccessful Queries: 1\nFailed Queries: 0\n")
            for suffix in module.SUFFIXES:
                (root / ("query_seed_42_sample_1" + suffix)).write_text("test prediction\n")

    def test_identical_outputs_pass(self):
        self.assertTrue(module.compare(self.a, self.b)["identical"])

    def test_timing_differences_are_excluded(self):
        (self.a / "timing.json").write_text('{"seconds":1}')
        (self.b / "timing.json").write_text('{"seconds":2}')
        self.assertTrue(module.compare(self.a, self.b)["identical"])

    def test_changed_prediction_fails(self):
        (self.b / "query_seed_42_sample_1_model.cif").write_text("different\n")
        self.assertFalse(module.compare(self.a, self.b)["identical"])

    def test_failed_query_is_rejected(self):
        (self.b / "summary.txt").write_text(
            "Total Queries Processed: 2\nSuccessful Queries: 1\nFailed Queries: 1\n")
        with self.assertRaises(ValueError):
            module.compare(self.a, self.b)

    def test_missing_confidence_is_rejected(self):
        (self.b / "query_seed_42_sample_1_confidences.json").unlink()
        with self.assertRaises(ValueError):
            module.compare(self.a, self.b)

    def test_empty_successful_directory_is_rejected(self):
        for path in self.b.glob("query*"):
            path.unlink()
        with self.assertRaises(ValueError):
            module.compare(self.a, self.b)

    def test_missing_summary_is_rejected(self):
        (self.b / "summary.txt").unlink()
        with self.assertRaises(OSError):
            module.compare(self.a, self.b)

    def test_changed_sample_set_fails(self):
        for path in list(self.b.glob("query*")):
            path.rename(path.with_name(path.name.replace("sample_1", "sample_2")))
        self.assertFalse(module.compare(self.a, self.b)["identical"])


if __name__ == "__main__":
    unittest.main()
