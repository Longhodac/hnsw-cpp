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
- Remaining: 4 SIMD/profiling, 5 concurrent inserts, 6 mmap persistence, 7 FAISS comparison. See README roadmap.
- Baseline to beat in Phase 4: the build (161 s at efC=100) and the QPS figures above, measured with the scalar `l2_sqr`.

## Known limitations to remember
- `search()` mutates the visited-tag scratch (`visited_`, `epoch_`), so it is not thread-safe; Phase 5 needs per-thread scratch.
- `search_layer` allocates two heaps per call; revisit when profiling in Phase 4.
- Tests reach private internals through the `HnswTestAccess` friend struct in `tests/hnsw_test_access.hpp`.
