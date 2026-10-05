#!/usr/bin/env python3
"""Memory, files and single-query latency for our index and FAISS, one library per process.

Two phases, each in a fresh process so nothing from the build leaks into the load numbers:

  build-save  build on --threads threads, record memory growth, save, record time and file size
  load        load that file, record load time, memory, first query, warm pass, and single-call
              latency at each ef (one query per call, so both libraries pay the same Python cost)

    .venv/bin/python scripts/bench_extras.py --lib ours  --phase build-save --dataset data/sift
    .venv/bin/python scripts/bench_extras.py --lib ours  --phase load       --dataset data/sift
    .venv/bin/python scripts/bench_extras.py --lib faiss --phase load --faiss-mmap --dataset data/sift
"""
import argparse
import csv
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from bench import BACKENDS
from bench_common import current_rss_mb, footprint_mb, load_dataset, recall_at_k, time_calls

COLUMNS = ["tag", "lib", "phase", "metric", "ef", "value", "unit"]


class Recorder:
    def __init__(self, args):
        self.args, self.rows = args, []

    def emit(self, metric, value, unit="", ef=""):
        # Print a readable number, but keep full precision in the CSV: rounding there once turned
        # recall 0.9513 into 0.951 and cut a 1.4 ms load time to 1 ms resolution.
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"{metric}{'@ef' + str(ef) if ef != '' else ''}={shown} {unit}".rstrip())
        stored = repr(value) if isinstance(value, float) else str(value)
        self.rows.append({"tag": self.args.tag, "lib": self.args.lib, "phase": self.args.phase,
                          "metric": metric, "ef": ef, "value": stored, "unit": unit})

    def flush(self):
        if self.args.csv == "none":
            return
        path = Path(self.args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if fresh:
                w.writeheader()
            w.writerows(self.rows)


def index_path(args):
    return Path(args.index_path or f"results/extras/{args.lib}_{Path(args.dataset).name}.idx")


def build_save(args, rec):
    data = load_dataset(args.dataset, args.max_queries)
    n, dim = data.base.shape
    backend = BACKENDS[args.lib](dim, n, args)
    fp0, rss0 = footprint_mb(), current_rss_mb()
    build_s = backend.build(data.base, args.threads)
    fp1, rss1 = footprint_mb(), current_rss_mb()
    rec.emit("build_s", build_s, "s")
    rec.emit("footprint_growth_after_build", fp1 - fp0, "MB")
    rec.emit("rss_growth_after_build", rss1 - rss0, "MB")

    path = index_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.lib == "ours":
        backend.index.save(str(path))
        np.save(str(path) + ".rows.npy", backend.row_of_id)  # ids were handed out by thread timing
    else:
        backend.faiss.write_index(backend.index, str(path))
    rec.emit("save_s", time.perf_counter() - t0, "s")
    rec.emit("file_mb", os.path.getsize(path) / 1e6, "MB")


class OursHandle:
    def __init__(self, path, args):
        sys.path.insert(0, args.module_dir)
        import hnsw_cpp

        t0 = time.perf_counter()
        self.index = hnsw_cpp.HnswIndex.load(str(path))
        self.load_s = time.perf_counter() - t0
        self.row_of_id = np.load(str(path) + ".rows.npy")

    def prepare(self, ef):
        pass

    def search(self, queries, k, ef):
        return self.index.search_batch(queries, k, ef, threads=1)[0]

    def to_rows(self, ids):
        return np.where(ids >= 0, self.row_of_id[np.maximum(ids, 0)], -1)


class FaissHandle:
    def __init__(self, path, args):
        import faiss

        self.faiss = faiss
        t0 = time.perf_counter()
        flags = faiss.IO_FLAG_MMAP if args.faiss_mmap else 0
        self.index = faiss.read_index(str(path), flags)
        self.load_s = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)

    def prepare(self, ef):
        self.index.hnsw.efSearch = ef

    def search(self, queries, k, ef):
        return self.index.search(queries, k)[1]

    def to_rows(self, ids):
        return ids


def percentile(sorted_values, p):
    """Nearest-rank percentile, the same definition as the C++ eval."""
    rank = min(max(math.ceil(p * len(sorted_values)), 1), len(sorted_values))
    return sorted_values[rank - 1]


def load_phase(args, rec):
    data = load_dataset(args.dataset, args.max_queries, with_base=False)
    nq = len(data.queries)
    path = index_path(args)
    rec.emit("file_mb", os.path.getsize(path) / 1e6, "MB")
    fp0, rss0 = footprint_mb(), current_rss_mb()
    try:
        handle = (OursHandle if args.lib == "ours" else FaissHandle)(path, args)
    except Exception as err:  # FAISS may refuse its mmap flag for an HNSW index
        rec.emit("load_failed", f"{type(err).__name__}: {err}")
        return
    rec.emit("load_s", handle.load_s, "s")
    rec.emit("footprint_growth_after_load", footprint_mb() - fp0, "MB")
    rec.emit("rss_growth_after_load", current_rss_mb() - rss0, "MB")

    ef0 = args.ef_search[0]
    handle.prepare(ef0)
    t0 = time.perf_counter()
    handle.search(data.queries[:1], args.k, ef0)
    rec.emit("first_query_us", (time.perf_counter() - t0) * 1e6, "us")

    seconds, ids = time_calls(lambda: handle.search(data.queries, args.k, ef0), repeats=1, warmup=0)
    rec.emit("first_full_pass_qps", nq / seconds[0], "qps")
    rec.emit("footprint_growth_after_pass", footprint_mb() - fp0, "MB")
    rec.emit("rss_growth_after_pass", current_rss_mb() - rss0, "MB")

    for ef in args.ef_search:
        handle.prepare(ef)
        rows = [data.queries[i:i + 1] for i in range(nq)]  # contiguous 1 x dim views, no copies
        for q in rows[:500]:  # warm-up
            handle.search(q, args.k, ef)
        out, ns = [], np.empty(nq, dtype=np.int64)
        gc.disable()
        for i, q in enumerate(rows):
            t0 = time.perf_counter_ns()
            out.append(handle.search(q, args.k, ef))
            ns[i] = time.perf_counter_ns() - t0
        gc.enable()
        recall = recall_at_k(handle.to_rows(np.concatenate(out)), data.gt, args.k)
        us = np.sort(ns) / 1e3
        rec.emit("recall", recall, "", ef)
        rec.emit("latency_p50_us", float(percentile(us, 0.50)), "us", ef)
        rec.emit("latency_p99_us", float(percentile(us, 0.99)), "us", ef)
        rec.emit("latency_mean_us", float(us.mean()), "us", ef)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--phase", required=True, choices=["build-save", "load"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index-path", default="")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=100)
    ap.add_argument("--threads", type=int, default=10, help="build threads")
    ap.add_argument("--ef-search", default="60,140", help="ef values for the latency runs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--faiss-mmap", action="store_true", help="read the FAISS index with IO_FLAG_MMAP")
    ap.add_argument("--module-dir", default="build/python/python")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default="results/extras.csv")
    args = ap.parse_args()
    args.ef_search = [int(x) for x in args.ef_search.split(",")]
    if args.faiss_mmap:
        args.phase_label = "load-mmap"

    rec = Recorder(args)
    print(f"lib={args.lib} phase={args.phase} dataset={args.dataset} M={args.M} efC={args.ef_construction}"
          f"{' faiss-mmap' if args.faiss_mmap else ''}")
    (build_save if args.phase == "build-save" else load_phase)(args, rec)
    rec.flush()


if __name__ == "__main__":
    main()
