"""Tests for scripts/bench_common.py, the loaders, recall and timing every benchmark shares."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench_common import current_rss_mb, peak_rss_mb, read_fvecs, read_ivecs, recall_at_k, time_calls  # noqa: E402


def write_vecs(path, matrix, dtype):
    """Write a TEXMEX file: each record is an int32 dim followed by the row's values."""
    n, dim = matrix.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = matrix.astype(dtype).view(np.int32)
    out.tofile(path)


class Loaders(unittest.TestCase):
    def test_fvecs_round_trip(self):
        matrix = np.random.default_rng(1).standard_normal((7, 5), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.fvecs"
            write_vecs(path, matrix, np.float32)
            got = read_fvecs(path)
        np.testing.assert_array_equal(got, matrix)
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(got.flags.c_contiguous)

    def test_ivecs_round_trip(self):
        matrix = np.arange(12, dtype=np.int32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.ivecs"
            write_vecs(path, matrix, np.int32)
            got = read_ivecs(path)
        np.testing.assert_array_equal(got, matrix)
        self.assertEqual(got.dtype, np.int32)

    def test_bad_files_raise_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.fvecs"
            empty.write_bytes(b"")
            with self.assertRaises(ValueError):
                read_fvecs(empty)

            truncated = Path(tmp) / "truncated.fvecs"
            write_vecs(truncated, np.ones((4, 6), dtype=np.float32), np.float32)
            truncated.write_bytes(truncated.read_bytes()[:-3])
            with self.assertRaises(ValueError):
                read_fvecs(truncated)

            mixed = Path(tmp) / "mixed.fvecs"
            rows = np.ones((3, 5), dtype=np.int32)
            rows[:, 0] = 4
            rows[1, 0] = 3  # one record claims a different dimension
            rows.tofile(mixed)
            with self.assertRaises(ValueError):
                read_fvecs(mixed)


class Recall(unittest.TestCase):
    gt = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

    def test_known_values(self):
        perfect = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])
        self.assertEqual(recall_at_k(perfect, self.gt, 4), 1.0)
        half = np.array([[1, 2, 99, 98], [5, 6, 97, 96]])
        self.assertEqual(recall_at_k(half, self.gt, 4), 0.5)
        self.assertEqual(recall_at_k(np.full((2, 4), 99), self.gt, 4), 0.0)

    def test_k_uses_only_the_first_k_of_each_side(self):
        labels = np.array([[1, 99, 98, 97], [5, 96, 95, 94]])
        self.assertEqual(recall_at_k(labels, self.gt, 1), 1.0)  # first result vs first truth
        self.assertEqual(recall_at_k(labels, self.gt, 2), 0.5)

    def test_faiss_padding_never_matches(self):
        self.assertEqual(recall_at_k(np.full((2, 4), -1), self.gt, 4), 0.0)

    def test_wider_label_matrix_is_cut_to_k(self):
        labels = np.array([[1, 2, 3, 4, 99, 98], [5, 6, 7, 8, 97, 96]])
        self.assertEqual(recall_at_k(labels, self.gt, 4), 1.0)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros((3, 4), dtype=np.int64), self.gt, 4)  # wrong query count
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros((2, 4), dtype=np.int64), self.gt, 0)
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros((2, 4), dtype=np.int64), self.gt, 5)  # wider than the truth


class Timing(unittest.TestCase):
    def test_warmup_and_repeat_counts_and_last_result(self):
        calls = []

        def fn():
            calls.append(1)
            return len(calls)

        seconds, result = time_calls(fn, repeats=3, warmup=2)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(seconds), 3)
        self.assertEqual(result, 5)
        self.assertTrue(all(s >= 0 for s in seconds))

    def test_memory_helpers_return_plausible_numbers(self):
        self.assertGreater(current_rss_mb(), 1)
        self.assertGreaterEqual(peak_rss_mb(), current_rss_mb() * 0.5)


if __name__ == "__main__":
    unittest.main()
