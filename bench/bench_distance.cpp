#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "hnsw/distance.hpp"

static void BM_L2Sqr(benchmark::State& state) {
    const auto dim = static_cast<size_t>(state.range(0));
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> a(dim), b(dim);
    for (auto& x : a) x = u(rng);
    for (auto& x : b) x = u(rng);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a.data());
        benchmark::DoNotOptimize(b.data());
        float d = hnsw::l2_sqr(a.data(), b.data(), dim);
        benchmark::DoNotOptimize(d);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(2 * dim * sizeof(float)));
}
BENCHMARK(BM_L2Sqr)->Arg(32)->Arg(96)->Arg(128)->Arg(768);
