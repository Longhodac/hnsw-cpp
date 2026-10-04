# hnsw-cpp

A from-scratch C++20 implementation of HNSW (Hierarchical Navigable Small World), written to learn systems programming. It is tested and benchmarked on the SIFT dataset, and a comparison against FAISS is the last roadmap item. The core library has no ANN or linear-algebra dependencies.

## HNSW in simple words

Paper: [Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*](https://arxiv.org/abs/1603.09320) (2016). Faiss, Weaviate, Qdrant, Milvus, Elasticsearch/OpenSearch and pgvector all offer HNSW. It serves semantic search, RAG retrieval and recommendations.

**The problem.** Given millions of items represented as vectors, find the ones most similar to a query. Comparing against every item is too slow, and exact tree-based shortcuts break down in high dimensions. Approximate nearest neighbor search accepts almost-best matches in exchange for a large speedup.

**The idea.** Finding an address, you start on a highway map with few points and long jumps, switch to a city map, then a street map. HNSW stores the data as a stack of proximity graphs.
- Layer 0 holds every item, linked to nearby neighbors.
- Higher layers hold exponentially fewer items, so their links span long distances. Each item's top layer is drawn at random, like a skip list over graphs.

**Search.** Start at the top layer and hop to whichever neighbor is closest to the query. When no neighbor is closer, drop a layer and continue from there. At layer 0, run a wider search of width `ef` and return the best K. The number of steps grows roughly with log(N), not N.

**Build.** Insert items one at a time. Pick a random top layer, search down from the top to find nearby nodes on each of the item's layers, then link the item to a selected few.

**The neighbor-selection heuristic.** Linking to the M closest nodes fails on clustered data, because all of them can sit in one cluster and the clusters never connect. The heuristic walks the candidates nearest-first and keeps one only if it is closer to the new item than to every neighbor already chosen. The kept links point in different directions, including across clusters.

| Parameter | Effect |
|---|---|
| `M` | Links per node. A higher value raises recall and memory use. |
| `efConstruction` | Search width while building. A higher value gives a better graph and a slower build. |
| `ef` (search) | Search width at query time. A higher value raises recall and lowers speed. This is the main speed/recall knob. |
| `mL` | Layer-height distribution. The paper suggests `1/ln(M)`. |

**Limitations noted by the authors.** HNSW uses more memory than compression-based methods such as product quantization. It is harder to distribute, since every search enters at the top layer. The original design has no deletes or updates. The log(N) bound is proven only under idealized assumptions, and the high-dimensional behavior rests on experiments.

## Layout

`include/hnsw` holds the public headers and `src` holds the library. `tools/eval.cpp` is the evaluation CLI. `tests` and `bench` hold the unit tests and microbenchmarks. `scripts` holds dataset download and plotting, and `data` holds the datasets and is gitignored.

## Build and test

You need CMake 3.24+, Ninja, and clang or gcc. CMake fetches GoogleTest and Google Benchmark.

```bash
cmake --preset release && cmake --build --preset release && ctest --preset release
```

The presets are `debug`, `release` (-O3, native CPU tuning), `asan` (ASan + UBSan) and `tsan`. Binaries land in `build/<preset>/` (`eval`, `hnsw_tests`, `bench_distance`).

## Dataset setup

```bash
scripts/download_sift.sh siftsmall   # 10K vectors, ~5 MB
scripts/download_sift.sh sift1m      # 1M vectors, ~160 MB download
```

Files go to `data/<name>/<name>_{base,query}.fvecs` and `<name>_groundtruth.ivecs`.

Run the evaluator and plot the results.

```bash
build/release/eval --dataset data/siftsmall --index bruteforce
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 100 --ef-search 10,20,50,100
scripts/plot_results.py results/results.csv      # recall@10 vs QPS (log) -> results/recall_qps.png
```

Each run appends rows to `results/results.csv`. The plot script draws one curve per `M` and `efConstruction` pair, so runs with the same pair but different seeds land on one zigzag curve. Keep them in separate CSVs.

## Roadmap

1. Harness: loaders, distance, brute force, recall, eval CLI, plotting, tests. Done.
2. Single-threaded HNSW. Done.
3. Parameter tuning of `M`, `efConstruction` and `ef`. Done, results below.
4. SIMD and profiling. NEON distance and prefetching done (about 2.8x faster queries). Heap work and memory layout still open.
5. Concurrent inserts and searches. Done: 5.4x faster build and 4.9x more QPS on 10 threads.
6. mmap persistence. Done: 1 ms load against a 69 s build.
7. FAISS comparison with pybind11 bindings.

## Phase 4: SIMD distance

Status: the NEON distance function and neighbor prefetching are done. The heap work and memory layout are not tried yet.

**What changed.** `include/hnsw/distance.hpp` now has three functions. `l2_sqr_scalar` is the old loop and stays as the reference for tests. `l2_sqr_neon` processes 16 floats per pass with four NEON accumulators and fused multiply-add, then a 4-wide loop, then a scalar loop for leftover elements. `l2_sqr` calls the NEON version on ARM and the scalar version elsewhere.

**Why it is faster.** SIMD means one instruction works on several numbers at once, and a NEON register holds 4 floats. At `-O3` the compiler already did the subtract and square with NEON, but it then added the 16 results one at a time in a single chain, because floating-point addition must keep its order. Each add waited for the one before it. The NEON version keeps four independent sums and combines them once at the end, so the adds overlap. Its distances can differ from the scalar version in the last digits.

**Profile before the change.** A 20-second sample of queries at `M=16`, `efC=100`, `ef=140` on SIFT1M, with `l2_sqr` marked non-inlined so it shows up on its own.

| Function | Share of samples |
|---|---|
| `l2_sqr` | 70.6% |
| `search_layer` itself | 25.6% |
| Heap pop | 3.2% |

**Results.** All numbers come from this Mac.

| Measurement | Scalar | NEON | Speedup |
|---|---|---|---|
| One distance call, 128 floats | 26.7 ns | 5.3 ns | 5.0x |
| One distance call, 768 floats | 294 ns | 34 ns | 8.6x |
| QPS at recall 0.95 (`ef` about 59) | about 7,700 | about 13,800 | 1.8x |
| QPS at recall 0.99 (`ef` about 158) | about 3,500 | about 5,900 | 1.7x |
| Build time, `M=16`, `efC=100` | 168 s and 180 s | 103 s and 104 s | 1.7x |

The QPS and build rows come from two scalar runs and two NEON runs made back to back with seed 42. Recall@10 was identical to four digits at every `ef`.

**Why 1.8x and not 5x.** Distance took 70.6% of query time. A 5x faster distance gives about 2.3x overall, and the measured gain was 1.8x. A second profile of the NEON build shows where the rest went.

| Function | Before NEON | After NEON |
|---|---|---|
| Distance | 70.6% | 37.3% |
| `search_layer` itself | 25.6% | 53.4% |
| Heap pop | 3.2% | 7.9% |
| `malloc` and `free` | 0.1% | 0.4% |

`malloc` and `free` are 0.4%, so reusing the two heaps would gain little. The rest of `search_layer` includes the visited-tag lookups and the neighbor-list reads, which are random reads from memory.

**Prefetching.** `search_layer` now makes one extra pass over a node's neighbors before the distance loop. It asks the CPU to start loading each neighbor's visited tag and vector, so the memory reads overlap instead of running one after another. Results do not change.

| Variant | QPS at recall 0.95 (`ef` 60) | QPS at `ef` 120 | Build |
|---|---|---|---|
| No prefetch | 13,800 | 7,900 | 99 s |
| Prefetch first cache line only | 15,100 | 8,800 | 88 s |
| Prefetch whole vector | 21,800 | 11,900 | 70 s |

Each row is two runs made back to back at `M=16`, `efC=100`, with identical recall at every `ef`. Prefetching the whole vector is about 1.6x faster at `ef=60`. The first-line version gains only about 10%, because a 512-byte vector spans four or five 128-byte cache lines and the later lines need their own requests. Runs at `ef` of 140 and 160 were noisy in every build, so the table uses lower values.

Together, NEON and prefetch give about 2.8x the QPS at recall 0.95 (7,700 to 21,800) and a 2.4x faster build (170 s to 70 s) compared with the Phase 3 scalar build. An early estimate put memory waits at about 16% of query time. Prefetch removed about 37%, so the waits were larger than estimated.

**How it was checked.** The unit tests compare the NEON and scalar functions for every dimension from 0 to 300 within a relative tolerance of 1e-5, and also cover unaligned pointers and an exact zero for identical vectors. The distance tests pass under AddressSanitizer, which catches reads past the end of an array. Recall at every `ef` matched the scalar build, and also matched with and without prefetch. The HNSW tests pass under AddressSanitizer with prefetching on.

## Phase 5: concurrent inserts and searches

Status: done. `add` and `search` are both safe to call from many threads at once.

**What changed.**
- Each thread keeps its own visited-tag scratch, so searches no longer share state.
- All storage is sized up front and never moves, and ids come from an atomic counter.
- Neighbor-list slots are atomic. Searches take no locks. An insert takes one of 4,096 striped mutexes (chosen by node id) while it rewrites a neighbor list, and never holds two at once, so the locks cannot deadlock.
- One global mutex covers the entry point and top layer. An insert takes it only when its node raises the top layer, which happens about log N times in a whole build.
- A node's level is a hash of the seed and its id, not a draw from a shared generator, so it does not depend on thread timing.

**How readers and writers stay safe.** A node's vector and link storage are written before its id goes into any neighbor slot. Slots are stored with release ordering and loaded with acquire ordering, so a thread that reads an id from a slot also sees that node's data. A search that overlaps a rewrite may read a mix of old and new ids from one list. Every id in a list is always a valid node on that layer, so the cost is at worst a skipped or repeated link.

**Results.** SIFT1M, `M=16`, `efC=100`, seed 42, on a Mac with 4 performance and 6 efficiency cores. Build rows are two runs per thread count. Query rows are two runs each over 20 passes of the query set.

| Build threads | Build time | Speedup | Recall@10 at `ef` 60 |
|---|---|---|---|
| 1 | 70.4 s | 1.0x | 0.9514 |
| 2 | 36.5 s | 1.9x | 0.9514 |
| 4 | 20.4 s | 3.5x | 0.9513 |
| 6 | 16.4 s | 4.3x | 0.9512 |
| 8 | 14.1 s | 5.0x | 0.9513 |
| 10 | 13.0 s | 5.4x | 0.9510 |

| Query threads | QPS at `ef` 60 | Speedup | p99 at `ef` 60 | QPS at `ef` 140 | Speedup |
|---|---|---|---|---|---|
| 1 | 21,400 | 1.0x | 64 µs | 10,400 | 1.0x |
| 2 | 41,100 | 1.9x | 68 µs | 20,000 | 1.9x |
| 4 | 72,700 | 3.4x | 101 µs | 35,600 | 3.4x |
| 6 | 88,000 | 4.1x | 114 µs | 43,400 | 4.2x |
| 8 | 99,800 | 4.7x | 192 µs | 49,100 | 4.7x |
| 10 | 105,500 | 4.9x | 297 µs | 51,900 | 5.0x |

Queries in the second table ran against a graph built by one thread, with recall 0.9514 at `ef` 60 and 0.9873 at `ef` 140 in every row.

**Reading the numbers.** Scaling is close to linear up to 4 threads, which matches the 4 performance cores. After that each extra thread adds less, because the slower efficiency cores join and the threads compete for memory bandwidth. Tail latency grows with it, from 64 µs at one thread to 297 µs at ten. Building with 10 threads is 13 seconds, down from 170 seconds with the Phase 3 scalar build.

Recall stays within 0.0004 of the single-thread build at every thread count. A parallel build gives a slightly different graph on every run, because the order in which threads insert nodes varies.

**A bug the sanitizers found.** In a parallel build, a node could end up linked to itself. Another thread could link a newer node to the new node on a lower layer first. The new node's own search then reached itself at distance 0 through that newer node and picked itself as a neighbor. A single thread cannot hit this. The insert now drops the new node from its own search results, and `add_link` refuses self-links. `ManySmallParallelBuildsStayValid` builds 40 small dense graphs with 8 threads and failed 20 times out of 20 with the fix removed.

**How it was checked.** All 52 tests pass in release and under both AddressSanitizer and ThreadSanitizer. The concurrency tests ran 200 times in release, 30 times under ASan and 15 times under TSan with no failures and no race reports. They cover parallel searches matching serial results, parallel builds producing a valid graph, frequent top-layer changes, searches running during inserts, and a parallel fill to exact capacity. Every test has a 300-second timeout, so a deadlock fails the test instead of hanging it.

**Run it.**

```bash
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 100 --threads 8 --ef-search 60,140
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 100 --ef-search 60 --search-threads 1,2,4,8 --query-repeat 20
```

`--threads` sets the build threads and `--search-threads` takes a list of query thread counts. `--query-repeat R` loops the query set R times so a fast multi-thread run lasts long enough to time. Recall and latency come from the first pass. The CSV gains `build_threads` and `search_threads` columns at the end, so use a fresh CSV file for runs with these options.

## Phase 6: saving and loading with mmap

Status: done. `HnswIndex::save(path)` writes the index to one file, and `HnswIndex::load(path)` maps it back in place. A loaded index is read-only, and `add` on it throws.

**How loading works.** `mmap` makes a file's bytes appear as ordinary memory. Nothing is copied or parsed at load. The operating system reads a 16 KiB page from disk the first time a search touches it, so load takes about a millisecond whatever the index size. Searches then run on the mapped bytes with the same code as an index built in memory.

**File format.** A 128-byte header, then five sections, each on a 128-byte boundary: vectors, node levels, layer 0 links, upper-layer offsets, and the upper-layer link arena. The header holds a magic string, a version, a byte-order marker, the parameters, the section offsets and a CRC-32 of itself. The format is defined in `include/hnsw/index_file.hpp`. For SIFT1M with `M=16` the file is 653.6 MB, 78% of it vectors.

**Storage changes this needed.** Layer 1 and above links used to live in a separate heap allocation per node, reached through a pointer, and pointers cannot go in a file. They now live in one flat arena, and each node stores an offset into it. A bump allocator hands out arena blocks during a build, with no lock. The arena is sized at 8 times the expected need as lazily committed memory, and a build throws `length_error` if it ever fills. Every array sits in a small `MemoryRegion` wrapper over `mmap`, and neighbor-list slots are plain `uint32_t` read and written through `std::atomic_ref`, so the same bytes work in memory and in a file.

**Results.** SIFT1M, `M=16`, `efC=100`, serial build, warm file cache.

| Measurement | Value |
|---|---|
| Build from scratch | 69.3 s |
| Save | 0.39 s, 653.6 MB |
| Load | 1.1 to 2.8 ms |
| Load with `verify` (checks every link) | 25.8 ms |
| Peak memory right after load | 18.7 MB |
| Peak memory after a full query pass | 677 MB |
| QPS at `ef` 60, loaded vs built in memory | 21,200 vs 20,900 |
| QPS at `ef` 140, loaded vs built in memory | 10,700 vs 10,700 |
| QPS with 10 query threads, `ef` 60 and 140 | 108,000 and 53,400 |
| Recall@10 at `ef` 60 and 140, loaded and built | 0.9514 and 0.9873, identical |

Loading is about 25,000 times faster than rebuilding, and the loaded index is as fast as the one built in memory once its pages are touched.

The first loaded run after the save was slower. Its first query took 81 ms and its first pass ran at 7,200 QPS with a p99 of 430 µs, against about 1 ms and 21,000 QPS in every later run. That fits part of the file not being resident yet. A truly cold cache needs `sudo purge` on macOS, which needs root and was not run, so cold-start numbers are not measured here.

**What load checks.** The default load checks the header, the byte order, the version, the header checksum, the section offsets against the file size, and that the entry node's level matches. That takes microseconds and reads no node data. It cannot see a damaged link, so only load files this program wrote. `LoadOptions{.verify = true}` also scans every link for out-of-range ids, self links, wrong layers and bad offsets, which cost 25.8 ms here with a warm cache.

**Limits.** A loaded index is read-only. The format is little-endian only, and loading uses POSIX `mmap`, so it targets macOS and Linux. The file stores no external ids, so `eval --save` writes `PATH.rows` next to a parallel-built index, which maps each id back to its base row.

**How it was checked.** All 72 tests pass in release and under both sanitizers. Round-trip tests compare 200 queries id for id against the original at three `ef` values, for serial-built, parallel-built, empty and single-node indexes. A table of 13 damaged files, including a bad magic, a future version, a flipped header bit, truncation and an inflated count, is each refused with a message that names the problem. A second table of 5 files with a damaged link inside a section loads under the default checks, and `verify` rejects each one. The save tests cover the exact file layout, overwriting an existing file, a missing directory, and no temporary file left behind.

**Run it.**

```bash
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 100 --threads 10 --ef-search 60 --save results/sift.idx
build/release/eval --dataset data/sift --index hnsw --load results/sift.idx --ef-search 60,60 --search-threads 1,10 --query-repeat 20
```

The first command builds in parallel and saves. The second loads without building and prints the load time, the first query's latency and the peak memory. `--verify 1` adds the link scan. Listing an `ef` twice shows the pass that pays for page faults next to the warm one.

## Benchmark results

### Setup

- Dataset: SIFT1M, 1,000,000 base vectors, 10,000 queries, 128 dimensions.
- Metric: recall@10 against the exact ground truth, and queries per second (QPS).
- Machine: one Mac with an M5 chip (4 performance and 6 efficiency cores), release preset.
- Seed 42 unless stated. Brute force reaches recall 0.9995 on this data, not 1.0, because of distance ties in the ground truth. Treat 0.9995 as the practical ceiling.
- QPS varies by 3% to 16% between runs of the same code (see Noise), so compare numbers only within one table.

### Summary

Every row uses `M=16`, `efConstruction=100` and `ef=60`, where recall@10 is about 0.951. Rows 1 to 3 come from back-to-back runs. Row 4 comes from the Phase 5 and Phase 6 runs, so its ratio to the other rows carries the usual run-to-run noise.

| Stage | QPS | Build time |
|---|---|---|
| Phase 3, scalar distance, 1 thread | about 7,700 | about 170 s |
| NEON distance, 1 thread | about 13,800 | about 104 s |
| NEON and prefetch, 1 thread | about 21,800 | about 70 s |
| NEON and prefetch, 10 query threads, 10 build threads | about 108,000 | about 13 s |
| Same index loaded from a file | same as built | 1.1 ms to load |

Compared with the Phase 3 build, one thread is about 2.8x faster at queries, ten threads are about 14x faster, and a ten-thread build takes 13 s instead of 170 s. A saved index loads in about a millisecond where a rebuild takes 69 s. Recall did not change at any stage. The phase sections above have the tables behind each row.

### Recommended setting

`M=16`, `efConstruction=100`, with the final build (NEON, prefetch).

| Target | `ef` | QPS, 1 thread | Mean latency | p99 latency |
|---|---|---|---|---|
| recall@10 of 0.95 | about 59 | about 22,000 | about 45 µs | about 65 µs |
| recall@10 of 0.99 | about 158 | about 9,500 | about 100 µs | about 135 µs |

A serial build takes about 69 s (68 s to 71 s over several runs), and a build on 10 threads takes about 13 s. A loaded index reaches 677 MB of resident memory after a full query pass. With 10 query threads, `ef` 60 reached 108,000 QPS and `ef` 140 reached 53,400 QPS (Phase 6 table).

The `ef` and QPS values come from the Phase 5 and Phase 6 runs on the final build, interpolated between `ef` 60, 140 and 160. In the Phase 3 sweep, `efConstruction=200` queried 3% to 10% faster than 100 but took 285 s to build with the scalar distance. That comparison was not repeated with NEON and prefetch.

### Parameter sweep

Phase 3 measurements with the scalar distance function, interpolated between the measured recall and QPS points. `ef` swept from 10 to 800 (or 400). Absolute QPS is about 2.8x lower than the final build, but the comparisons between rows hold.

| `M` | `efConstruction` | Build | `ef` at 0.95 | QPS at 0.95 | `ef` at 0.99 | QPS at 0.99 |
|---|---|---|---|---|---|---|
| 16 | 40 | 72 s | 105 | 6,000 | 366 | 2,200 |
| 16 | 100 | 188 s | 59 | 7,900 | 165 | 3,600 |
| 16 | 200 | 285 s | 53 | 9,500 | 132 | 4,400 |
| 16 | 400 | 618 s | 51 | 8,100 | 127 | 4,000 |
| 8 | 200 | 230 s | 118 | 6,600 | 375 | 2,200 |
| 32 | 200 | 437 s | 36 | 7,600 | 91 | 3,900 |
| 48 | 200 | 426 s | 34 | 8,600 | 83 | 4,300 |

The `efConstruction=100` row comes from a first run that was slower for reasons outside the code. A back-to-back rerun against `efConstruction=200` gave about 9,100 and 4,000 QPS for it (see Noise).

![Recall@10 vs QPS for each M and efConstruction pair on SIFT1M](docs/phase3_recall_qps.png)

Each curve is one graph with `ef` swept along it. A curve that sits further up and to the right is better. The dip in the `M=16, efConstruction=400` curve near recall 0.9 is a noisy run, not a real effect (see Noise).

### What the sweeps show

- **`ef`.** At `M=16`, `efConstruction=200`, recall rises from 0.71 at `ef=10` to 0.946 at `ef=50` and 0.9956 at `ef=200`. It reaches 0.9993 at `ef=800`. Mean latency grows by about 1.3 µs per unit of `ef`, and p99 stays at 1.3 to 1.5 times the mean.
- **`efConstruction`.** Build time scales almost directly with it (72 s, 188 s, 285 s, 618 s). Going from 40 to 100 cuts the `ef` needed for recall 0.95 from 105 to 59. Going from 200 to 400 doubles the build time and gains no QPS. At `efConstruction=40` the graph tops out near recall 0.992 even at `ef=400`.
- **`M`.** `M=8` has the lowest QPS and the lowest recall ceiling, and saves only about 20% of the build time. `M=16`, 32 and 48 reach about the same QPS at equal recall, within noise. Larger `M` needs a smaller `ef` but spends more time on each hop, and it costs more build time and memory, so `M=16` wins.
- **Heuristic ablation.** Not run yet. The `--heuristic 0` flag turns the diversity heuristic off, and a run at `M=16`, `efConstruction=100` would show whether it helps on SIFT.

### Noise

- The seed changes recall by 0.0003 or less. Seed 42 and seed 7 at `M=16`, `efConstruction=200` reach 0.9585 and 0.9588 at `ef=60`.
- QPS for the same configuration varies by 3% to 16% between runs, depending on what the machine is doing. The `efConstruction=100` run at `ef=60` gave 7,742 QPS the first time and 8,994 QPS the second time, at the same recall.
- QPS can even fall out of order inside one run. One `efConstruction=400` run shows 7,200 QPS at `ef=30`, below the 9,141 QPS at `ef=40`.
- Treat QPS gaps under about 15% as noise unless a repeat run confirms them, and compare configurations from runs made back to back.

### Reproduce

```bash
build/release/eval --dataset data/sift --index hnsw --M 16 --ef-construction 100 --ef-search 10,20,30,40,50,55,60,70,80,100,120,140,160,200,400 --csv results/sweep.csv
scripts/plot_results.py results/sweep.csv -o docs/phase3_recall_qps.png
```

The plot lives in `docs/` because `results/` and `*.png` are gitignored. `.gitignore` has an exception for `docs/*.png`.
