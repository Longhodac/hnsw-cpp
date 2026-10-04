#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "hnsw/distance.hpp"

namespace {

template <float (*Fn)(const float*, const float*, size_t)>
void run_l2(benchmark::State& state) {
    const auto dim = static_cast<size_t>(state.range(0));
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> a(dim), b(dim);
    for (auto& x : a) x = u(rng);
    for (auto& x : b) x = u(rng);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a.data());
        benchmark::DoNotOptimize(b.data());
        float d = Fn(a.data(), b.data(), dim);
        benchmark::DoNotOptimize(d);
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(2 * dim * sizeof(float)));
}

}  // namespace

static void BM_L2SqrScalar(benchmark::State& state) { run_l2<hnsw::l2_sqr_scalar>(state); }
BENCHMARK(BM_L2SqrScalar)->Arg(32)->Arg(96)->Arg(128)->Arg(768);

#if defined(HNSW_HAS_NEON)
static void BM_L2SqrNeon(benchmark::State& state) { run_l2<hnsw::l2_sqr_neon>(state); }
BENCHMARK(BM_L2SqrNeon)->Arg(32)->Arg(96)->Arg(128)->Arg(768);
#endif
