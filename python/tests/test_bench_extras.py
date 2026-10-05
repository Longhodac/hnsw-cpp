"""Tests for the pieces of scripts/bench_extras.py that do not need an index."""
import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench_extras import Recorder, percentile  # noqa: E402


class RecorderTests(unittest.TestCase):
    def make(self, csv_path):
        return Recorder(Namespace(tag="A", lib="ours", phase="load", csv=csv_path))

    def test_csv_keeps_full_float_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            rec = self.make(str(path))
            rec.emit("recall", 0.951337, "", 60)
            rec.emit("load_s", 0.0014271, "s")
            rec.flush()
            rows = list(csv.DictReader(path.open()))
        self.assertEqual(float(rows[0]["value"]), 0.951337)  # not rounded to 0.951
        self.assertEqual(float(rows[1]["value"]), 0.0014271)  # not rounded to 0.001
        self.assertEqual((rows[0]["tag"], rows[0]["lib"], rows[0]["phase"], rows[0]["ef"]), ("A", "ours", "load", "60"))

    def test_strings_are_stored_as_is_and_header_is_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            for _ in range(2):  # two processes appending to one file
                rec = self.make(str(path))
                rec.emit("load_failed", "RuntimeError: refused")
                rec.flush()
            lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "tag,lib,phase,metric,ef,value,unit")
        self.assertEqual(sum(1 for line in lines if line.startswith("tag,")), 1)
        self.assertEqual(len(lines), 3)

    def test_csv_none_writes_nothing(self):
        rec = self.make("none")
        rec.emit("x", 1.0)
        rec.flush()  # must not raise or create a file


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_matches_the_cpp_definition(self):
        values = list(range(1, 101))  # 1..100
        self.assertEqual(percentile(values, 0.50), 50)
        self.assertEqual(percentile(values, 0.99), 99)
        self.assertEqual(percentile(values, 1.00), 100)
        self.assertEqual(percentile([7.0], 0.99), 7.0)
        self.assertEqual(percentile(values, 0.0), 1)  # the rank is clamped to the first element


if __name__ == "__main__":
    unittest.main()
