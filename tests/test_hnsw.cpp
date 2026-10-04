#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "hnsw/brute_force.hpp"
#include "hnsw/dataset.hpp"
#include "hnsw/distance.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/metrics.hpp"
#include "hnsw_test_access.hpp"

using hnsw::HnswIndex;
using hnsw::HnswTestAccess;

namespace {

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

// Mean recall@k of `idx` vs exact search on `queries`.
double recall_vs_exact(const hnsw::Index& idx, const std::vector<float>& base, size_t dim,
                       const std::vector<float>& queries, size_t k, size_t ef) {
    hnsw::BruteForceIndex exact(dim);
    for (size_t i = 0; i < base.size() / dim; ++i) exact.add(&base[i * dim]);
    size_t hits = 0, total = 0;
    for (size_t q = 0; q < queries.size() / dim; ++q) {
        std::set<uint32_t> truth;
        for (auto& [d, id] : exact.search(&queries[q * dim], k, 0)) truth.insert(id);
        for (auto& [d, id] : idx.search(&queries[q * dim], k, ef)) hits += truth.count(id);
        total += k;
    }
    return static_cast<double>(hits) / static_cast<double>(total);
}

}  // namespace

TEST(Hnsw, EmptyAndSingle) {
    HnswIndex idx(3, 8, 50, 10, 1);
    const float q[] = {0, 0, 0};
    EXPECT_TRUE(idx.search(q, 5, 10).empty());

    const float v[] = {1, 1, 1};
    EXPECT_EQ(idx.add(v), 0u);
    const auto r = idx.search(q, 5, 10);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].second, 0u);
    EXPECT_FLOAT_EQ(r[0].first, 3.f);
}

TEST(Hnsw, ThrowsWhenFull) {
    HnswIndex idx(1, 4, 20, 2, 1);
    const float v[] = {0};
    idx.add(v);
    idx.add(v);
    EXPECT_THROW(idx.add(v), std::length_error);
}

TEST(Hnsw, ResultsWellFormed) {
    const size_t n = 1000, dim = 8;
    const auto base = random_vectors(n, dim, 1);
    HnswIndex idx(dim, 12, 100, n, 42);
    for (size_t i = 0; i < n; ++i) EXPECT_EQ(idx.add(&base[i * dim]), i);
    HnswTestAccess::validate(idx);
    EXPECT_GE(HnswTestAccess::max_level(idx), 1);  // 1000 nodes, M=12: levels > 0 exist

    const auto q = random_vectors(1, dim, 2);
    const auto r = idx.search(q.data(), 10, 50);
    ASSERT_EQ(r.size(), 10u);
    std::set<uint32_t> ids;
    for (size_t i = 0; i < r.size(); ++i) {
        EXPECT_LT(r[i].second, n);
        EXPECT_TRUE(ids.insert(r[i].second).second) << "duplicate id";
        EXPECT_FLOAT_EQ(r[i].first, hnsw::l2_sqr(q.data(), &base[r[i].second * dim], dim));
        if (i) EXPECT_LE(r[i - 1].first, r[i].first);
    }
}

TEST(Hnsw, KLargerThanSize) {
    HnswIndex idx(2, 4, 20, 10, 1);
    for (int i = 0; i < 3; ++i) {
        const float v[] = {static_cast<float>(i), 0};
        idx.add(v);
    }
    const float q[] = {0, 0};
    EXPECT_EQ(idx.search(q, 10, 10).size(), 3u);
}

TEST(Hnsw, DuplicateVectors) {
    HnswIndex idx(2, 4, 20, 50, 1);
    const float v[] = {1, 1};
    for (int i = 0; i < 50; ++i) idx.add(v);
    HnswTestAccess::validate(idx);
    EXPECT_EQ(idx.search(v, 5, 20).size(), 5u);
}

TEST(Hnsw, NearExactAtLargeEf) {
    const size_t n = 2000, dim = 16;
    const auto base = random_vectors(n, dim, 3);
    const auto queries = random_vectors(100, dim, 4);
    HnswIndex idx(dim, 16, 200, n, 42);
    for (size_t i = 0; i < n; ++i) idx.add(&base[i * dim]);
    EXPECT_GE(recall_vs_exact(idx, base, dim, queries, 10, n), 0.99);
}

TEST(Hnsw, RecallGrowsWithEf) {
    const size_t n = 3000, dim = 16;
    const auto base = random_vectors(n, dim, 5);
    const auto queries = random_vectors(100, dim, 6);
    HnswIndex idx(dim, 8, 100, n, 42);
    for (size_t i = 0; i < n; ++i) idx.add(&base[i * dim]);
    const double lo = recall_vs_exact(idx, base, dim, queries, 10, 10);
    const double hi = recall_vs_exact(idx, base, dim, queries, 10, 200);
    EXPECT_LE(lo, hi);
    EXPECT_GE(hi, 0.95);
}

TEST(Hnsw, SameSeedSameGraph) {
    const size_t n = 500, dim = 8;
    const auto base = random_vectors(n, dim, 7);
    HnswIndex a(dim, 8, 50, n, 9), b(dim, 8, 50, n, 9);
    for (size_t i = 0; i < n; ++i) {
        a.add(&base[i * dim]);
        b.add(&base[i * dim]);
    }
    const auto q = random_vectors(1, dim, 8);
    EXPECT_EQ(a.search(q.data(), 10, 50), b.search(q.data(), 10, 50));
}

TEST(Hnsw, SiftSmallRecall) {
    const std::filesystem::path dir = std::filesystem::path(HNSW_DATA_DIR) / "siftsmall";
    if (!std::filesystem::exists(dir / "siftsmall_base.fvecs")) {
        GTEST_SKIP() << "run scripts/download_sift.sh siftsmall first";
    }
    const auto base = hnsw::load_fvecs((dir / "siftsmall_base.fvecs").string());
    const auto query = hnsw::load_fvecs((dir / "siftsmall_query.fvecs").string());
    const auto gt = hnsw::load_ivecs((dir / "siftsmall_groundtruth.ivecs").string());

    for (const bool heuristic : {true, false}) {
        HnswIndex idx(base.dim, 16, 200, base.count, 42);
        idx.set_use_heuristic(heuristic);
        for (size_t i = 0; i < base.count; ++i) idx.add(base.row(i));
        HnswTestAccess::validate(idx);

        std::vector<std::vector<uint32_t>> res(query.count);
        for (size_t q = 0; q < query.count; ++q) {
            for (auto& [d, id] : idx.search(query.row(q), 10, 100)) res[q].push_back(id);
        }
        EXPECT_GE(hnsw::recall_at_k(res, gt, 10), 0.95) << "heuristic=" << heuristic;
    }
}

namespace {

std::unique_ptr<HnswIndex> build_index(const std::vector<float>& base, size_t dim,
                                       uint64_t seed) {
    auto idx = std::make_unique<HnswIndex>(dim, 12, 100, base.size() / dim, seed);
    for (size_t i = 0; i < base.size() / dim; ++i) idx->add(&base[i * dim]);
    return idx;
}

using Results = std::vector<std::vector<hnsw::Neighbor>>;

Results run_queries(const HnswIndex& idx, const std::vector<float>& queries, size_t dim, size_t k,
                    size_t ef) {
    Results out;
    for (size_t q = 0; q < queries.size() / dim; ++q) {
        out.push_back(idx.search(&queries[q * dim], k, ef));
    }
    return out;
}

}  // namespace

TEST(HnswConcurrency, ParallelSearchMatchesSerial) {
    const size_t n = 3000, dim = 16, k = 10, ef = 60;
    const auto base = random_vectors(n, dim, 3);
    const auto queries = random_vectors(200, dim, 4);
    const auto built = build_index(base, dim, 42);
    const HnswIndex& idx = *built;
    const Results serial = run_queries(idx, queries, dim, k, ef);

    constexpr size_t kThreads = 4;
    std::vector<Results> got(kThreads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < 3; ++round) got[t] = run_queries(idx, queries, dim, k, ef);
        });
    }
    for (auto& th : threads) th.join();
    for (size_t t = 0; t < kThreads; ++t) EXPECT_EQ(got[t], serial) << "thread " << t;
}

TEST(HnswConcurrency, TwoIndexesOnOneThreadDoNotInterfere) {
    const size_t dim = 8, k = 5, ef = 40;
    const auto base_a = random_vectors(500, dim, 5);
    const auto base_b = random_vectors(800, dim, 6);  // different size forces a scratch reset
    const auto queries = random_vectors(50, dim, 7);
    const auto built_a = build_index(base_a, dim, 1);
    const auto built_b = build_index(base_b, dim, 2);
    const HnswIndex& a = *built_a;
    const HnswIndex& b = *built_b;
    const Results want_a = run_queries(a, queries, dim, k, ef);
    const Results want_b = run_queries(b, queries, dim, k, ef);

    for (int round = 0; round < 3; ++round) {  // alternate on this thread
        for (size_t q = 0; q < queries.size() / dim; ++q) {
            EXPECT_EQ(a.search(&queries[q * dim], k, ef), want_a[q]);
            EXPECT_EQ(b.search(&queries[q * dim], k, ef), want_b[q]);
        }
    }
}

namespace {

// Vectors of `idx` in id order, so exact search runs over exactly what the index holds
// (a parallel build assigns ids in a thread-dependent order).
std::vector<float> stored_vectors(const HnswIndex& idx, size_t dim) {
    std::vector<float> out(idx.size() * dim);
    for (uint32_t id = 0; id < idx.size(); ++id) {
        std::copy_n(HnswTestAccess::vec(idx, id), dim, out.begin() + static_cast<long>(id * dim));
    }
    return out;
}

void add_in_parallel(HnswIndex& idx, const std::vector<float>& base, size_t dim, size_t threads) {
    const size_t n = base.size() / dim;
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (size_t i = t; i < n; i += threads) idx.add(&base[i * dim]);
        });
    }
    for (auto& th : pool) th.join();
}

}  // namespace

TEST(HnswConcurrency, ParallelBuildProducesValidGraph) {
    const size_t n = 3000, dim = 16;
    const auto base = random_vectors(n, dim, 11);
    const auto queries = random_vectors(100, dim, 12);

    HnswIndex serial(dim, 12, 100, n, 42);
    for (size_t i = 0; i < n; ++i) serial.add(&base[i * dim]);
    const double serial_recall = recall_vs_exact(serial, base, dim, queries, 10, 100);

    HnswIndex parallel(dim, 12, 100, n, 42);
    add_in_parallel(parallel, base, dim, 4);
    ASSERT_EQ(parallel.size(), n);
    HnswTestAccess::validate(parallel);
    const double parallel_recall =
        recall_vs_exact(parallel, stored_vectors(parallel, dim), dim, queries, 10, 100);
    EXPECT_GT(parallel_recall, 0.9);
    EXPECT_GT(parallel_recall, serial_recall - 0.05);
}

TEST(HnswConcurrency, FrequentTopLevelChangesStayValid) {
    // M=3 gives many layers, so inserts often raise the top layer and fight for entry_mutex_.
    const size_t n = 2000, dim = 8;
    const auto base = random_vectors(n, dim, 13);
    HnswIndex idx(dim, 3, 60, n, 5);
    add_in_parallel(idx, base, dim, 8);
    ASSERT_EQ(idx.size(), n);
    HnswTestAccess::validate(idx);
    EXPECT_GE(HnswTestAccess::max_level(idx), 3);
    const auto r = idx.search(&base[0], 5, 50);
    EXPECT_EQ(r.size(), 5u);
}

TEST(HnswConcurrency, SearchWhileAdding) {
    const size_t preload = 500, total = 2000, dim = 12, k = 10;
    const auto base = random_vectors(total, dim, 21);
    const auto queries = random_vectors(64, dim, 22);
    HnswIndex idx(dim, 8, 80, total, 9);
    for (size_t i = 0; i < preload; ++i) idx.add(&base[i * dim]);

    std::atomic<bool> writers_done{false};
    std::atomic<size_t> bad{0}, searches{0};
    std::vector<std::thread> pool;
    for (size_t t = 0; t < 2; ++t) {
        pool.emplace_back([&, t] {  // two writers split the remaining vectors
            for (size_t i = preload + t; i < total; i += 2) idx.add(&base[i * dim]);
        });
    }
    std::vector<std::thread> readers;
    for (size_t t = 0; t < 2; ++t) {
        readers.emplace_back([&, t] {
            size_t q = t;
            while (!writers_done.load()) {
                const auto r = idx.search(&queries[(q % 64) * dim], k, 40);
                ++q;
                ++searches;
                std::set<uint32_t> seen;
                for (size_t i = 0; i < r.size(); ++i) {
                    if (r[i].second >= total || !seen.insert(r[i].second).second) ++bad;
                    if (i > 0 && r[i].first < r[i - 1].first) ++bad;
                }
                if (r.empty() || r.size() > k) ++bad;
            }
        });
    }
    for (auto& th : pool) th.join();
    writers_done = true;
    for (auto& th : readers) th.join();

    EXPECT_EQ(bad.load(), 0u);
    EXPECT_GT(searches.load(), 0u);
    ASSERT_EQ(idx.size(), total);
    HnswTestAccess::validate(idx);
}

TEST(HnswConcurrency, ParallelAddStopsAtCapacity) {
    const size_t cap = 400, dim = 4, threads = 4;
    const auto base = random_vectors(cap, dim, 31);
    HnswIndex idx(dim, 6, 40, cap, 3);
    std::atomic<size_t> ok{0}, full{0};
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&] {
            for (size_t i = 0; i < cap / 2; ++i) {  // 4 * 200 attempts for 400 slots
                try {
                    idx.add(&base[(i % cap) * dim]);
                    ++ok;
                } catch (const std::length_error&) {
                    ++full;
                }
            }
        });
    }
    for (auto& th : pool) th.join();
    EXPECT_EQ(ok.load(), cap);
    EXPECT_EQ(full.load(), threads * (cap / 2) - cap);
    ASSERT_EQ(idx.size(), cap);
    HnswTestAccess::validate(idx);
}

TEST(HnswConcurrency, ManySmallParallelBuildsStayValid) {
    // Small dense graphs with many threads make inserts collide most often: threads find each
    // other's half-linked nodes, including a node reaching itself through a newer neighbor.
    const size_t n = 200, dim = 4;
    for (uint64_t round = 0; round < 40; ++round) {
        const auto base = random_vectors(n, dim, static_cast<uint32_t>(100 + round));
        HnswIndex idx(dim, 4, 30, n, round);
        add_in_parallel(idx, base, dim, 8);
        ASSERT_EQ(idx.size(), n) << "round " << round;
        HnswTestAccess::validate(idx);
        if (::testing::Test::HasFailure()) FAIL() << "invalid graph in round " << round;
    }
}
