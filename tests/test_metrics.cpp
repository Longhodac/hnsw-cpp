#include <gtest/gtest.h>

#include "hnsw/metrics.hpp"

TEST(Metrics, RecallPartial) {
    const hnsw::IvecsData gt{3, 2, {1, 2, 3, 4, 5, 6}};
    // q0 finds 2 of its 3 true neighbors; q1 finds all 3.
    const std::vector<std::vector<uint32_t>> r = {{1, 2, 9}, {6, 5, 4}};
    EXPECT_NEAR(hnsw::recall_at_k(r, gt, 3), 5.0 / 6.0, 1e-12);
}

TEST(Metrics, RecallUsesOnlyFirstK) {
    const hnsw::IvecsData gt{3, 1, {1, 2, 3}};
    EXPECT_DOUBLE_EQ(hnsw::recall_at_k({{1, 3, 2}}, gt, 1), 1.0);
    EXPECT_DOUBLE_EQ(hnsw::recall_at_k({{3, 2, 1}}, gt, 1), 0.0);
}

TEST(Metrics, RecallValidatesInput) {
    const hnsw::IvecsData gt{3, 1, {1, 2, 3}};
    EXPECT_THROW(hnsw::recall_at_k({}, gt, 1), std::invalid_argument);
    EXPECT_THROW(hnsw::recall_at_k({{1}}, gt, 4), std::invalid_argument);
}

TEST(Metrics, LatencyPercentiles) {
    std::vector<double> v;
    for (int i = 1; i <= 100; ++i) v.push_back(i);
    const auto s = hnsw::summarize_latencies(v);
    EXPECT_DOUBLE_EQ(s.mean_us, 50.5);
    EXPECT_DOUBLE_EQ(s.p50_us, 50);
    EXPECT_DOUBLE_EQ(s.p99_us, 99);
}
