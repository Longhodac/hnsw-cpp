#!/usr/bin/env python3
"""FAISS HNSW on a TEXMEX dataset: build, sweep ef, report recall and QPS.

The baseline for the comparison with our index. It uses the same loader, recall code and
timing helper as the later shared harness, so a harness bug shows up here first.

    .venv/bin/python scripts/faiss_baseline.py --dataset data/siftsmall
    .venv/bin/python scripts/faiss_baseline.py --dataset data/sift --threads 10 \\
        --ef-search 10,20,30,50,60,80,100,140,200,400,800
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import faiss
import numpy as np

from bench_common import load_dataset, peak_rss_mb, recall_at_k, time_calls

COLUMNS = ["lib", "dataset", "n", "nq", "dim", "k", "M", "ef_construction", "threads", "ef_search",
           "recall", "qps_median", "qps_min", "qps_max", "build_s", "repeats", "peak_rss_mb"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads for build and search")
    ap.add_argument("--ef-search", default="10,20,30,50,60,80,100,140,200,400,800")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="timed search passes per ef")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--csv", default="results/faiss_baseline.csv", help="'none' to skip")
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.max_queries)
    n, dim = data.base.shape
    nq = data.queries.shape[0]
    faiss.omp_set_num_threads(args.threads)
    print(f"faiss {faiss.__version__} omp_threads={faiss.omp_get_max_threads()} "
          f"options={faiss.get_compile_options()}")
    print(f"dataset={data.name} n={n} nq={nq} dim={dim} M={args.M} efC={args.ef_construction} "
          f"threads={args.threads} k={args.k}")

    index = faiss.IndexHNSWFlat(dim, args.M)
    index.hnsw.efConstruction = args.ef_construction
    t0 = time.perf_counter()
    index.add(data.base)
    build_s = time.perf_counter() - t0
    print(f"build_s={build_s:.3f} ntotal={index.ntotal}")

    rows = []
    for ef in (int(x) for x in args.ef_search.split(",")):
        index.hnsw.efSearch = ef
        seconds, (_, labels) = time_calls(lambda: index.search(data.queries, args.k), args.repeats)
        qps = [nq / s for s in seconds]
        recall = recall_at_k(labels, data.gt, args.k)
        print(f"ef_search={ef} recall@{args.k}={recall:.4f} qps_median={statistics.median(qps):.1f} "
              f"min={min(qps):.1f} max={max(qps):.1f}")
        rows.append({"lib": "faiss", "dataset": data.name, "n": n, "nq": nq, "dim": dim, "k": args.k,
                     "M": args.M, "ef_construction": args.ef_construction, "threads": args.threads,
                     "ef_search": ef, "recall": f"{recall:.6f}", "qps_median": f"{statistics.median(qps):.2f}",
                     "qps_min": f"{min(qps):.2f}", "qps_max": f"{max(qps):.2f}", "build_s": f"{build_s:.4f}",
                     "repeats": args.repeats, "peak_rss_mb": f"{peak_rss_mb():.1f}"})
    print(f"peak_rss_mb={peak_rss_mb():.1f}")

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
