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
- Phase 1 (harness) and Phase 2 (single-threaded HNSW) are done. SIFT1M: recall@10 0.974 at ~5.8K QPS (M=16, efC=200, ef=80), build ~329 s.
- Brute-force recall on SIFT1M is ~0.9995, not 1.0: distance ties vs the ground truth, not a bug.
- Remaining: 3 tuning sweeps, 4 SIMD/profiling, 5 concurrent inserts, 6 mmap persistence, 7 FAISS comparison. See README roadmap.

## Known limitations to remember
- `search()` mutates the visited-tag scratch (`visited_`, `epoch_`), so it is not thread-safe; Phase 5 needs per-thread scratch.
- `search_layer` allocates two heaps per call; revisit when profiling in Phase 4.
- Tests reach private internals through the `HnswTestAccess` friend struct in `tests/hnsw_test_access.hpp`.
