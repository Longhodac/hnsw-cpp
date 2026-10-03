# hnsw-cpp

## Overview
A from-scratch C++20 implementation of HNSW (Malkov & Yashunin, 2016), built as a
systems-programming learning project and benchmarked against FAISS on SIFT.
The core library has no ANN or linear-algebra dependencies.

Layout: `include/hnsw` (public headers), `src` (library), `tests`, `bench`,
`tools/eval.cpp` (evaluation CLI), `scripts` (dataset download, plotting), `data` (gitignored).

## Build and test
Requires CMake 3.24+, Ninja, and clang or gcc. GoogleTest and Google Benchmark are
fetched automatically.

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

Presets: `debug`, `release` (-O3, native CPU tuning), `asan` (ASan + UBSan), `tsan`.
Binaries land in `build/<preset>/` (`eval`, `hnsw_tests`, `bench_distance`).

## Dataset setup
```bash
scripts/download_sift.sh siftsmall   # 10K vectors, ~5 MB
scripts/download_sift.sh sift1m      # 1M vectors, ~160 MB download
```
Files go to `data/<name>/<name>_{base,query}.fvecs` and `<name>_groundtruth.ivecs`.

Run the evaluator and plot:
```bash
build/release/eval --dataset data/siftsmall --index bruteforce
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 200 --ef-search 10,20,50,100
scripts/plot_results.py results/results.csv      # recall@10 vs QPS (log) -> results/recall_qps.png
```
Each run appends rows to `results/results.csv`.

## Roadmap
1. **Harness**: loaders, distance, brute force, recall, eval CLI, plotting, tests
2. Single-threaded HNSW
3. Parameter tuning (M, efConstruction, efSearch)
4. SIMD and profiling
5. Concurrent inserts
6. mmap persistence
7. FAISS comparison with pybind11 bindings

## Benchmark results
_TBD_
