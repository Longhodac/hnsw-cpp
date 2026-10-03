# hnsw-cpp

A from-scratch C++20 implementation of **HNSW** (Hierarchical Navigable Small World), built to learn systems programming and benchmarked against FAISS on the SIFT dataset.

## HNSW in simple words

Paper: [Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*](https://arxiv.org/abs/1603.09320) (2016). This algorithm is what powers or is offered by many of today's vector databases and search systems, including Faiss, Weaviate, Qdrant, Milvus, Elasticsearch/OpenSearch and pgvector. It underlies semantic search, RAG retrieval and recommendations.

**The problem.** Given millions of items represented as vectors, find the ones most similar to a query. Comparing against every item is too slow, and exact tree-based shortcuts break down in high dimensions. So we use *approximate* nearest neighbor search: accept almost-best matches in exchange for a huge speedup.

**The idea: a map with zoom levels.** Finding an address, you start on a highway map (few points, long jumps), switch to a city map, then a street map. HNSW stores the data as a stack of proximity graphs:
- Layer 0 holds every item, linked to nearby neighbors.
- Higher layers hold exponentially fewer items, so their links span long distances. Each item's top layer is drawn at random, like a skip list but over graphs.

**Search.** Start at the top layer and greedily hop to whichever neighbor is closest to the query. When no neighbor is closer, drop a layer and continue from there. At layer 0, run a wider search (width `ef`) and return the best K. Steps grow roughly with log(N), not N.

**Build.** Insert items one at a time: pick a random top layer, search down from the top to find nearby nodes at each of its layers, then link it to a selected few.

**The neighbor-selection heuristic.** Linking to the M closest nodes fails on clustered data, because all of them can sit in one cluster and the clusters never connect. The heuristic walks candidates nearest-first and keeps one only if it is closer to the new item than to any neighbor already chosen. That yields links in diverse directions, including bridges between clusters.

| Parameter | Effect |
|---|---|
| `M` | Links per node. Higher means better recall and more memory. |
| `efConstruction` | Search width while building. Higher means a better graph and slower build. |
| `ef` (search) | Search width at query time. Higher means more accurate and slower. The main speed/recall knob. |
| `mL` | Layer-height distribution; the paper suggests `1/ln(M)`. |

**Limitations noted by the authors.** High memory use compared with compression-based methods (e.g. product quantization); harder to distribute, since every search enters at the top layer; no deletes or updates in the original design; and the log(N) bound is proven only under idealized assumptions, with high-dimensional behavior backed by experiments.

**What this repo does.** Phase 1 (the evaluation harness) is done. Phases 2 onward implement HNSW and measure it with the same harness (see the roadmap below).

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
