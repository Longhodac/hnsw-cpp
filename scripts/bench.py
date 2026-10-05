#!/usr/bin/env python3
"""One benchmark for both libraries: our HNSW (hnsw_cpp) and FAISS's IndexHNSWFlat.

Both go through the same data loading, timer, recall code and CSV format. Only the library
call differs, and only that call is timed. Run each library in its own process.

    .venv/bin/python scripts/bench.py --lib ours  --dataset data/sift --threads 10
    .venv/bin/python scripts/bench.py --lib faiss --dataset data/sift --threads 10

Threads apply to the build and to the search. --search-threads overrides the search side.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from bench_common import current_rss_mb, load_dataset, peak_rss_mb, recall_at_k, time_calls

COLUMNS = ["lib", "dataset", "n", "nq", "dim", "k", "M", "ef_construction", "build_threads",
           "search_threads", "ef_search", "recall", "qps_median", "qps_min", "qps_max",
           "qps_spread", "build_s", "repeats", "rss_build_mb", "peak_rss_mb"]
NOISE_MARGIN = 0.15  # repeats that differ by more than this are flagged


class OursBackend:
    name = "ours"

    def __init__(self, dim, n, args):
        sys.path.insert(0, args.module_dir)
        import hnsw_cpp

        self.index = hnsw_cpp.HnswIndex(dim, n, M=args.M, ef_construction=args.ef_construction,
                                        seed=args.seed)
        self.row_of_id = None
        self.version = f"hnsw_cpp module at {args.module_dir}"

    def build(self, base, threads):
        t0 = time.perf_counter()
        ids = self.index.add_batch(base, threads=threads)
        seconds = time.perf_counter() - t0
        # The index hands out its own ids. Map them back to base rows once, outside the timer.
        self.row_of_id = np.empty(len(base), dtype=np.int64)
        self.row_of_id[ids] = np.arange(len(base))
        return seconds

    def make_search(self, queries, k, ef, threads):
        return lambda: self.index.search_batch(queries, k, ef, threads=threads)

    def labels(self, raw):
        ids = raw[0]
        return np.where(ids >= 0, self.row_of_id[np.maximum(ids, 0)], -1)


class FaissBackend:
    name = "faiss"

    def __init__(self, dim, n, args):
        import faiss

        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(dim, args.M)
        self.index.hnsw.efConstruction = args.ef_construction
        self.version = f"faiss {faiss.__version__} ({faiss.get_compile_options()})"

    def build(self, base, threads):
        self.faiss.omp_set_num_threads(threads)
        t0 = time.perf_counter()
        self.index.add(base)
        return time.perf_counter() - t0

    def make_search(self, queries, k, ef, threads):
        self.faiss.omp_set_num_threads(threads)
        self.index.hnsw.efSearch = ef
        return lambda: self.index.search(queries, k)

    def labels(self, raw):
        return raw[1]  # FAISS ids are base rows already, -1 where it found fewer than k


BACKENDS = {"ours": OursBackend, "faiss": FaissBackend}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1, help="threads for the build and the search")
    ap.add_argument("--search-threads", type=int, default=0, help="override threads for the search")
    ap.add_argument("--ef-search", default="10,20,30,50,60,80,100,140,200,400,800")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="timed search passes per ef")
    ap.add_argument("--warmup", type=int, default=1, help="untimed search passes per ef")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="our index only; FAISS has its own")
    ap.add_argument("--module-dir", default="build/python/python", help="where hnsw_cpp*.so is")
    ap.add_argument("--csv", default="results/bench.csv", help="'none' to skip")
    args = ap.parse_args()
    search_threads = args.search_threads or args.threads

    data = load_dataset(args.dataset, args.max_queries)
    n, dim = data.base.shape
    nq = data.queries.shape[0]
    backend = BACKENDS[args.lib](dim, n, args)
    print(f"lib={args.lib} {backend.version}")
    print(f"dataset={data.name} n={n} nq={nq} dim={dim} M={args.M} efC={args.ef_construction} "
          f"build_threads={args.threads} search_threads={search_threads} k={args.k}")

    rss_before = current_rss_mb()
    build_s = backend.build(data.base, args.threads)
    rss_build = current_rss_mb() - rss_before  # growth around the build, without the dataset
    print(f"build_s={build_s:.3f} rss_build_mb={rss_build:.1f}")

    rows = []
    for ef in (int(x) for x in args.ef_search.split(",")):
        run = backend.make_search(data.queries, args.k, ef, search_threads)
        seconds, raw = time_calls(run, args.repeats, args.warmup)
        labels = backend.labels(raw)
        recall = recall_at_k(labels, data.gt, args.k)
        qps = [nq / s for s in seconds]
        median = statistics.median(qps)
        spread = (max(qps) - min(qps)) / median
        flag = "  NOISY" if spread > NOISE_MARGIN else ""
        print(f"ef_search={ef} recall@{args.k}={recall:.4f} qps_median={median:.1f} "
              f"min={min(qps):.1f} max={max(qps):.1f} spread={spread:.1%}{flag}")
        rows.append({"lib": args.lib, "dataset": data.name, "n": n, "nq": nq, "dim": dim, "k": args.k,
                     "M": args.M, "ef_construction": args.ef_construction,
                     "build_threads": args.threads, "search_threads": search_threads,
                     "ef_search": ef, "recall": f"{recall:.6f}", "qps_median": f"{median:.2f}",
                     "qps_min": f"{min(qps):.2f}", "qps_max": f"{max(qps):.2f}",
                     "qps_spread": f"{spread:.4f}", "build_s": f"{build_s:.4f}",
                     "repeats": args.repeats, "rss_build_mb": f"{rss_build:.1f}",
                     "peak_rss_mb": f"{peak_rss_mb():.1f}"})

    if args.csv != "none":
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if fresh:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
