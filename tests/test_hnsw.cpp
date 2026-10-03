#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <set>

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
