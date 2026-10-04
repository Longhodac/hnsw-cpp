"""Tests for the hnsw_cpp Python module. Run with: ctest --preset python

The module must be on PYTHONPATH (the python preset's ctest entry sets it).
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

import hnsw_cpp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_common import load_dataset, recall_at_k  # noqa: E402


def random_matrix(n, dim, seed):
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def built_index(n=1500, dim=16, M=12, seed=42, threads=1):
    base = random_matrix(n, dim, seed)
    index = hnsw_cpp.HnswIndex(dim, n, M=M, ef_construction=100, seed=seed)
    ids = index.add_batch(base, threads=threads)
    return index, base, ids


class Basics(unittest.TestCase):
    def test_properties_and_repr(self):
        index = hnsw_cpp.HnswIndex(8, 100, M=6)
        self.assertEqual((index.dim, index.size, index.capacity, index.read_only), (8, 0, 100, False))
        self.assertEqual(len(index), 0)
        self.assertIn("dim=8", repr(index))

    def test_bad_constructor_arguments_raise_value_error(self):
        with self.assertRaises(ValueError):
            hnsw_cpp.HnswIndex(0, 10)
        with self.assertRaises(ValueError):
            hnsw_cpp.HnswIndex(4, 10, M=1)

    def test_serial_add_returns_row_order_ids(self):
        index, base, ids = built_index(300, 8)
        self.assertEqual(ids.dtype, np.uint32)
        np.testing.assert_array_equal(ids, np.arange(300))
        self.assertEqual(index.size, 300)

    def test_parallel_add_returns_a_permutation_of_ids(self):
        index, base, ids = built_index(1200, 8, threads=4)
        self.assertEqual(index.size, 1200)
        np.testing.assert_array_equal(np.sort(ids), np.arange(1200))

    def test_each_row_finds_itself_after_a_parallel_build(self):
        index, base, ids = built_index(1200, 8, threads=4)
        found, dists = index.search_batch(base, k=1, ef=50)
        self.assertGreater(float((found[:, 0] == ids).mean()), 0.98)
        self.assertTrue((dists[found[:, 0] == ids] == 0).all())


class Search(unittest.TestCase):
    def test_result_shapes_dtypes_and_order(self):
        index, base, _ = built_index()
        queries = random_matrix(40, 16, 7)
        ids, dists = index.search_batch(queries, k=10, ef=50)
        self.assertEqual((ids.shape, dists.shape), ((40, 10), (40, 10)))
        self.assertEqual((ids.dtype, dists.dtype), (np.int64, np.float32))
        self.assertTrue((np.diff(dists, axis=1) >= 0).all(), "nearest first")
        self.assertTrue(((ids >= 0) & (ids < 1500)).all())

    def test_threads_do_not_change_results(self):
        index, base, _ = built_index()
        queries = random_matrix(300, 16, 8)
        one = index.search_batch(queries, k=10, ef=60, threads=1)
        for threads in (2, 4, 7):
            many = index.search_batch(queries, k=10, ef=60, threads=threads)
            np.testing.assert_array_equal(many[0], one[0])
            np.testing.assert_array_equal(many[1], one[1])

    def test_index_smaller_than_k_pads_with_minus_one_and_infinity(self):
        index = hnsw_cpp.HnswIndex(4, 10)
        index.add_batch(random_matrix(2, 4, 1))
        ids, dists = index.search_batch(random_matrix(3, 4, 2), k=5, ef=10)
        self.assertTrue((ids[:, 2:] == -1).all())
        self.assertTrue(np.isinf(dists[:, 2:]).all())
        self.assertTrue((ids[:, :2] >= 0).all())

    def test_empty_index_returns_only_padding(self):
        index = hnsw_cpp.HnswIndex(4, 10)
        ids, dists = index.search_batch(random_matrix(3, 4, 2), k=3, ef=10)
        self.assertTrue((ids == -1).all() and np.isinf(dists).all())

    def test_matches_the_cpp_eval_on_siftsmall(self):
        directory = ROOT / "data" / "siftsmall"
        if not directory.exists():
            self.skipTest("data/siftsmall is not downloaded")
        data = load_dataset(directory)
        index = hnsw_cpp.HnswIndex(data.base.shape[1], len(data.base), M=16, ef_construction=100, seed=42)
        index.add_batch(data.base)  # one thread: same graph as the C++ eval with seed 42
        ids, _ = index.search_batch(data.queries, k=10, ef=60)
        # build/release/eval --dataset data/siftsmall --index hnsw --M 16 --ef-construction 100
        # --ef-search 60 prints recall@10=0.9970
        self.assertAlmostEqual(recall_at_k(ids, data.gt, 10), 0.9970, places=4)


class InputChecks(unittest.TestCase):
    def setUp(self):
        self.index, self.base, _ = built_index(200, 8)

    def assertValueError(self, fn, *needles):
        with self.assertRaises(ValueError) as caught:
            fn()
        for needle in needles:
            self.assertIn(needle, str(caught.exception))

    def test_float64_is_refused_not_converted(self):
        self.assertValueError(lambda: self.index.search_batch(self.base.astype(np.float64), 5, 10), "float32", "float64")
        self.assertValueError(lambda: self.index.add_batch(self.base.astype(np.float64)), "float32")

    def test_non_contiguous_is_refused(self):
        wide = random_matrix(50, 16, 3)
        self.assertValueError(lambda: self.index.search_batch(wide[:, ::2], 5, 10), "C-contiguous")
        self.assertValueError(lambda: self.index.search_batch(self.base.T.copy().T, 5, 10), "C-contiguous")

    def test_wrong_rank_and_wrong_dim_are_refused(self):
        self.assertValueError(lambda: self.index.search_batch(self.base[0], 5, 10), "2-dimensional")
        self.assertValueError(lambda: self.index.search_batch(random_matrix(4, 9, 1), 5, 10), "9 columns", "dimension 8")

    def test_bad_k_ef_threads_are_refused(self):
        self.assertValueError(lambda: self.index.search_batch(self.base, 0, 10), "k must")
        self.assertValueError(lambda: self.index.search_batch(self.base, 5, 0), "ef must")
        self.assertValueError(lambda: self.index.search_batch(self.base, 5, 10, threads=0), "threads")
        self.assertValueError(lambda: self.index.add_batch(self.base, threads=0), "threads")

    def test_exceeding_capacity_adds_nothing(self):
        index = hnsw_cpp.HnswIndex(8, 100)
        index.add_batch(random_matrix(60, 8, 1))
        self.assertValueError(lambda: index.add_batch(random_matrix(50, 8, 2)), "exceed the capacity of 100", "nothing was added")
        self.assertEqual(index.size, 60)
        index.add_batch(random_matrix(40, 8, 3))  # exactly fills it
        self.assertEqual(index.size, 100)


class Persistence(unittest.TestCase):
    def test_round_trip_gives_identical_results_and_is_read_only(self):
        index, base, _ = built_index(1200, 16)
        queries = random_matrix(100, 16, 9)
        want = index.search_batch(queries, k=10, ef=60)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "index.bin")
            index.save(path)
            loaded = hnsw_cpp.HnswIndex.load(path)
            self.assertTrue(loaded.read_only)
            self.assertEqual((loaded.size, loaded.dim), (index.size, index.dim))
            got = loaded.search_batch(queries, k=10, ef=60, threads=3)
            np.testing.assert_array_equal(got[0], want[0])
            np.testing.assert_array_equal(got[1], want[1])
            with self.assertRaises(RuntimeError) as caught:
                loaded.add_batch(random_matrix(1, 16, 1))
            self.assertIn("read-only", str(caught.exception))
            hnsw_cpp.HnswIndex.load(path, verify=True)  # a good file passes the full check

    def test_loading_a_missing_or_damaged_file_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                hnsw_cpp.HnswIndex.load(str(Path(tmp) / "missing.bin"))
            bad = Path(tmp) / "bad.bin"
            bad.write_bytes(b"not an index" * 20)
            with self.assertRaises(RuntimeError) as caught:
                hnsw_cpp.HnswIndex.load(str(bad))
            self.assertIn("bad magic", str(caught.exception))


class Threading(unittest.TestCase):
    def test_python_threads_can_search_one_index_at_once(self):
        index, base, _ = built_index(2000, 16)
        queries = random_matrix(200, 16, 11)
        want = index.search_batch(queries, k=10, ef=60)
        got = [None] * 4

        def work(slot):
            for _ in range(3):
                got[slot] = index.search_batch(queries, k=10, ef=60, threads=2)

        pool = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        for slot in range(4):
            np.testing.assert_array_equal(got[slot][0], want[0])

    def test_search_releases_the_gil(self):
        # A pure-Python thread records the longest gap between two of its own iterations. While
        # search_batch holds the GIL the spinner cannot run, so that gap is as long as the whole
        # call. With the GIL released the gap stays near the scheduler's few milliseconds.
        index, base, _ = built_index(5000, 32)
        queries = np.ascontiguousarray(base)
        while True:  # make one call long enough to measure
            t0 = time.perf_counter()
            index.search_batch(queries, k=10, ef=200, threads=1)
            duration = time.perf_counter() - t0
            if duration > 0.15:
                break
            queries = np.ascontiguousarray(np.tile(queries, (2, 1)))

        stop = threading.Event()
        longest_gap = [0.0]

        def spin():
            last = time.perf_counter()
            while not stop.is_set():
                now = time.perf_counter()
                longest_gap[0] = max(longest_gap[0], now - last)
                last = now

        spinner = threading.Thread(target=spin)
        spinner.start()
        time.sleep(0.05)
        longest_gap[0] = 0.0  # ignore the start-up
        t0 = time.perf_counter()
        index.search_batch(queries, k=10, ef=200, threads=1)
        duration = time.perf_counter() - t0
        stall = longest_gap[0]
        stop.set()
        spinner.join()
        self.assertLess(stall, duration / 2,
                        f"the spinner stalled {stall * 1000:.0f} ms of a {duration * 1000:.0f} ms call, "
                        "so the GIL was held during search")


if __name__ == "__main__":
    unittest.main()
