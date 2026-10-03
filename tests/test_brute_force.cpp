#include <gtest/gtest.h>

#include <filesystem>

#include "hnsw/brute_force.hpp"
#include "hnsw/dataset.hpp"
#include "hnsw/metrics.hpp"

TEST(BruteForce, SmallExample) {
    hnsw::BruteForceIndex idx(2);
    const float pts[][2] = {{0, 0}, {1, 0}, {5, 5}, {0, 2}};
    for (const auto& p : pts) idx.add(p);
    EXPECT_EQ(idx.size(), 4u);

    const float q[] = {0.1f, 0.f};
    const auto r = idx.search(q, 3, 0);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].second, 0u);
    EXPECT_EQ(r[1].second, 1u);
    EXPECT_EQ(r[2].second, 3u);
    EXPECT_LE(r[0].first, r[1].first);
    EXPECT_LE(r[1].first, r[2].first);
}

TEST(BruteForce, KLargerThanSize) {
    hnsw::BruteForceIndex idx(1);
    const float a[] = {1}, b[] = {2};
    idx.add(a);
    idx.add(b);
    EXPECT_EQ(idx.search(a, 10, 0).size(), 2u);
}

TEST(BruteForce, TiesBreakByLowerId) {
    hnsw::BruteForceIndex idx(1);
    const float p[] = {3};
    for (int i = 0; i < 5; ++i) idx.add(p);
    const auto r = idx.search(p, 2, 0);
    EXPECT_EQ(r[0].second, 0u);
    EXPECT_EQ(r[1].second, 1u);
}

TEST(BruteForce, PerfectRecallOnSiftSmall) {
    const std::filesystem::path dir = std::filesystem::path(HNSW_DATA_DIR) / "siftsmall";
    if (!std::filesystem::exists(dir / "siftsmall_base.fvecs")) {
        GTEST_SKIP() << "run scripts/download_sift.sh siftsmall first";
    }
    const auto base = hnsw::load_fvecs((dir / "siftsmall_base.fvecs").string());
    const auto query = hnsw::load_fvecs((dir / "siftsmall_query.fvecs").string());
    const auto gt = hnsw::load_ivecs((dir / "siftsmall_groundtruth.ivecs").string());

    hnsw::BruteForceIndex idx(base.dim);
    for (size_t i = 0; i < base.count; ++i) idx.add(base.row(i));

    std::vector<std::vector<uint32_t>> results(query.count);
    for (size_t q = 0; q < query.count; ++q) {
        for (const auto& [d, id] : idx.search(query.row(q), 10, 0)) results[q].push_back(id);
    }
    EXPECT_DOUBLE_EQ(hnsw::recall_at_k(results, gt, 10), 1.0);
}
