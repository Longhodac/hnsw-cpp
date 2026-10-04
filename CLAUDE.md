# Agent instructions

This is a learning project. For HNSW logic, concurrency, and SIMD code, give hints, point out bugs, and review the user's code rather than writing full implementations, unless the user explicitly asks for code.

When the user explicitly says to "implement" or "build" something, write the complete working implementation with tests and don't hold back. Without that explicit request, keep to hints and review.

## Project notes
- C++20, CMake presets: `debug`, `release`, `asan`, `tsan` (`cmake --preset X && cmake --build --preset X`).
- Core library has no third-party dependencies; GoogleTest/Google Benchmark are fetched for tests/bench only.
- Vectors live in one flat `std::vector<float>`; nodes are `uint32_t` ids.
- `include/hnsw/index.hpp` is the common interface; `HnswIndex` (`src/hnsw_index.cpp`) implements it.
- Scaffolding, harness, tests, scripts, and docs are fair game to write directly.
- Use explicit `std::` qualifiers; never `using namespace std;`.

## Status
- Phases 1 (harness), 2 (single-threaded HNSW) and 3 (parameter tuning) are done. Results and tables are in the README "Benchmark results" section.
- Recommended setting: M=16, efC=100. SIFT1M recall@10 0.95 at ef about 59 (about 9.1K QPS) and 0.99 at ef about 159 (about 4.0K QPS), build 161 s. efC=200 is about 3% to 10% faster on queries but builds in 285 s. efC=400 and M above 16 gain nothing, M=8 is worse.
- Noise: seed changes recall by at most 0.0003. QPS for one config varies 3% to 16% between runs, so treat gaps under about 15% as noise and compare back-to-back runs.
- Brute-force recall on SIFT1M is ~0.9995, not 1.0: distance ties vs the ground truth, not a bug.
- Heuristic ablation (`--heuristic 0`) was not run. Optional follow-up.
- Phase 4 (SIMD): `l2_sqr` now uses NEON (`l2_sqr_neon`, 4 accumulators, 16 floats per pass); `l2_sqr_scalar` stays as the test reference. Distance call 26.7 ns to 5.3 ns at dim 128. End to end at M=16, efC=100: about 1.8x QPS (13.8K at recall 0.95, 5.9K at 0.99) and build 170 s to 104 s, recall unchanged. Details in the README.
- Phase 3 QPS figures above were measured with the scalar distance, so they are the pre-Phase-4 baseline.
- Prefetch: `search_layer` prefetches each neighbor's visited tag and whole vector (128-byte lines) before the distance loop. A/B at M=16, efC=100: about 1.6x QPS at ef=60 (13.8K to 21.8K), build 99 s to 70 s, recall identical. Prefetching only the first line gains about 10%. NEON plus prefetch is about 2.8x QPS and 2.4x build speed versus the Phase 3 scalar build.
- Phase 4 still open: re-profile the prefetch build. Candidates are the heap pop (about 8% before prefetch), the visited-tag layout, and memory layout.
- Phase 5 plan (agreed): 5a per-thread search scratch (done), 5b preallocated storage and atomic id counter (done), 5c per-node-striped locks (done) (4096 mutexes by id) with atomic neighbor slots for lock-free readers, global mutex for entry point and top level only when a new top level appears, level from hash(seed, id), 5d mixed search-while-add stress test under TSan, 5e `eval --threads N` scaling. Each step ends in a check. 5a to 5c: 52 tests pass in release, ASan and TSan; concurrency tests repeated 200x release, 30x ASan, 15x TSan with no failures; single-thread QPS (about 22.5K at ef=60) and recall (within 0.0004) unchanged. 5c found and fixed a race where a node reached itself through a newer neighbor and linked to itself (see `add()`); `ManySmallParallelBuildsStayValid` fails 20/20 without the fix. Still open: 5d (mostly covered by `SearchWhileAdding`) and 5e (`eval --threads N`, scaling measurement).
- `HnswIndex` is neither copyable nor movable now (atomics, mutexes). Build it with `make_unique` or in place. `random_level(id)` is a hash of (seed, id), not a stream, so single-thread graphs differ from pre-5c ones.
- Remaining after that: 6 mmap persistence, 7 FAISS comparison. See README roadmap.

## Known limitations to remember
- `search()` is thread-safe (per-thread `thread_local` visited scratch, Phase 5a). `add()` is thread-safe too (Phase 5c): neighbor slots are atomic (release stores, acquire loads), 4096 striped per-node mutexes (one held at a time), and `entry_mutex_` only when a node raises the top layer. A parallel build gives a different graph each run.
- `search_layer` allocates two heaps per call, but `malloc` and `free` are only about 0.4% of query time, so reusing them gains little.
- Tests reach private internals through the `HnswTestAccess` friend struct in `tests/hnsw_test_access.hpp`.
