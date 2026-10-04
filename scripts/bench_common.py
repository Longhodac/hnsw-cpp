"""Shared pieces of the Python benchmarks: TEXMEX loaders, recall and timing.

The recall definition matches src/metrics.cpp, so a number from Python and a number from the
C++ eval CLI mean the same thing.
"""
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_vecs(path: Path, dtype) -> np.ndarray:
    """Read a TEXMEX .fvecs/.ivecs file: records of [int32 dim][dim 4-byte values]."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path}: file is empty")
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise ValueError(f"{path}: size does not fit records of dimension {dim}")
    rows = raw.reshape(-1, dim + 1)
    if not (rows[:, 0] == dim).all():
        raise ValueError(f"{path}: records have inconsistent dimensions")
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)  # C-contiguous copy, reinterpreted


def read_fvecs(path) -> np.ndarray:
    return _read_vecs(Path(path), np.float32)


def read_ivecs(path) -> np.ndarray:
    return _read_vecs(Path(path), np.int32)


@dataclass
class Dataset:
    name: str
    base: np.ndarray  # (n, dim) float32, C-contiguous
    queries: np.ndarray  # (nq, dim) float32, C-contiguous
    gt: np.ndarray  # (nq, width) int32, true neighbors by base row


def load_dataset(directory, max_queries: int = 0) -> Dataset:
    """Load <dir>/<name>_{base,query}.fvecs and <name>_groundtruth.ivecs, name = dir basename."""
    d = Path(directory)
    name = d.name or d.parent.name
    base = read_fvecs(d / f"{name}_base.fvecs")
    queries = read_fvecs(d / f"{name}_query.fvecs")
    gt = read_ivecs(d / f"{name}_groundtruth.ivecs")
    if queries.shape[1] != base.shape[1]:
        raise ValueError(f"query dim {queries.shape[1]} != base dim {base.shape[1]}")
    if gt.shape[0] != queries.shape[0]:
        raise ValueError(f"ground truth has {gt.shape[0]} rows but there are {queries.shape[0]} queries")
    if max_queries and max_queries < queries.shape[0]:
        queries, gt = queries[:max_queries], gt[:max_queries]
    return Dataset(name, base, np.ascontiguousarray(queries), gt)


def recall_at_k(labels: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of the first k results found among the true k nearest neighbors.

    `labels` is (nq, >=k) of base-row ids; an id of -1 (FAISS padding) never matches.
    """
    if labels.shape[0] != gt.shape[0]:
        raise ValueError(f"labels has {labels.shape[0]} queries but ground truth has {gt.shape[0]}")
    if k < 1 or k > gt.shape[1]:
        raise ValueError(f"k={k} must be in [1, ground-truth width {gt.shape[1]}]")
    found = (labels[:, :k, None] == gt[:, None, :k]).any(axis=2)
    return float(found.sum()) / (gt.shape[0] * k)


def time_calls(fn, repeats: int = 3, warmup: int = 1):
    """Run fn() warmup times, then `repeats` timed times. Returns (seconds per repeat, last result)."""
    for _ in range(warmup):
        fn()
    seconds, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - t0)
    return seconds, result


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far. ru_maxrss is bytes on macOS, KB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
