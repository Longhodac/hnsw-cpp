# Agent instructions

This is a learning project. For HNSW logic, concurrency, and SIMD code, give hints, point out bugs, and review the user's code rather than writing full implementations, unless the user explicitly asks for code.

## Project notes
- C++20, CMake presets: `debug`, `release`, `asan`, `tsan` (`cmake --preset X && cmake --build --preset X`).
- Core library has no third-party dependencies; GoogleTest/Google Benchmark are fetched for tests/bench only.
- Vectors live in one flat `std::vector<float>`; nodes are `uint32_t` ids.
- `include/hnsw/index.hpp` is the common interface; `HnswIndex` (src/hnsw_index.cpp) is the user's to implement.
- Scaffolding, harness, tests, scripts, and docs are fair game to write directly.
