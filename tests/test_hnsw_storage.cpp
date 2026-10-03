#include <gtest/gtest.h>

#include <vector>

#include "hnsw_test_access.hpp"

using hnsw::HnswIndex;
using hnsw::HnswTestAccess;

TEST(HnswStorage, DegreeLimits) {
    HnswIndex idx(2, 4, 10, 8, 1);
    EXPECT_EQ(HnswTestAccess::max_degree(idx, 0), 8u);  // 2M
    EXPECT_EQ(HnswTestAccess::max_degree(idx, 1), 4u);  // M
}

TEST(HnswStorage, AllocateStoresVectorsAndStartsEmpty) {
    HnswIndex idx(2, 4, 10, 8, 1);
    const float a[] = {1, 2}, b[] = {3, 4};
    EXPECT_EQ(HnswTestAccess::allocate(idx, a, 0), 0u);
    EXPECT_EQ(HnswTestAccess::allocate(idx, b, 2), 1u);
    EXPECT_EQ(idx.size(), 2u);
    EXPECT_EQ(HnswTestAccess::vec(idx, 1)[1], 4.f);
    EXPECT_TRUE(HnswTestAccess::nbrs(idx, 0, 0).empty());
    for (int l = 0; l <= 2; ++l) EXPECT_TRUE(HnswTestAccess::nbrs(idx, 1, l).empty());
}

TEST(HnswStorage, LayersDoNotOverlap) {
    HnswIndex idx(1, 3, 10, 8, 1);
    const float v[] = {0};
    const auto a = HnswTestAccess::allocate(idx, v, 2);  // layers 0,1,2
    const auto b = HnswTestAccess::allocate(idx, v, 1);  // layers 0,1
    const auto c = HnswTestAccess::allocate(idx, v, 3);  // layers 0..3

    HnswTestAccess::set(idx, a, 0, {1, 2, 3, 4, 5, 6});  // full layer 0 (2M = 6)
    HnswTestAccess::set(idx, a, 1, {7, 8, 9});            // full upper layer (M = 3)
    HnswTestAccess::set(idx, a, 2, {10});
    HnswTestAccess::set(idx, b, 1, {11, 12, 13});
    HnswTestAccess::set(idx, c, 1, {21});
    HnswTestAccess::set(idx, c, 3, {22, 23});
    HnswTestAccess::set(idx, b, 0, {99});

    auto eq = [](std::span<const uint32_t> s, std::vector<uint32_t> want) {
        return std::vector<uint32_t>(s.begin(), s.end()) == want;
    };
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, a, 0), {1, 2, 3, 4, 5, 6}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, a, 1), {7, 8, 9}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, a, 2), {10}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, b, 0), {99}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, b, 1), {11, 12, 13}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, c, 1), {21}));
    EXPECT_TRUE(HnswTestAccess::nbrs(idx, c, 2).empty());
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, c, 3), {22, 23}));

    // Shrinking a list must not disturb neighbors' blocks.
    HnswTestAccess::set(idx, a, 0, {42});
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, a, 0), {42}));
    EXPECT_TRUE(eq(HnswTestAccess::nbrs(idx, b, 0), {99}));
}

TEST(HnswStorage, FullIndexThrows) {
    HnswIndex idx(1, 2, 10, 2, 1);
    const float v[] = {0};
    HnswTestAccess::allocate(idx, v, 0);
    HnswTestAccess::allocate(idx, v, 0);
    EXPECT_THROW(HnswTestAccess::allocate(idx, v, 0), std::length_error);
}

TEST(HnswStorage, RejectsBadParams) {
    EXPECT_THROW(HnswIndex(0, 4, 10, 8, 1), std::invalid_argument);
    EXPECT_THROW(HnswIndex(2, 1, 10, 8, 1), std::invalid_argument);
}

TEST(HnswLevels, DeterministicForSeed) {
    HnswIndex a(2, 16, 10, 8, 7), b(2, 16, 10, 8, 7);
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(HnswTestAccess::random_level(a), HnswTestAccess::random_level(b));
    }
}

TEST(HnswLevels, GeometricDistribution) {
    const size_t M = 16;
    HnswIndex idx(2, M, 10, 8, 123);
    const int n = 200000;
    std::vector<int> hist(32, 0);
    for (int i = 0; i < n; ++i) ++hist[std::min(HnswTestAccess::random_level(idx), 31)];

    // P(level >= l) = M^-l.
    int at_least_1 = n - hist[0];
    EXPECT_NEAR(static_cast<double>(at_least_1) / n, 1.0 / M, 0.004);
    int at_least_2 = at_least_1 - hist[1];
    EXPECT_NEAR(static_cast<double>(at_least_2) / n, 1.0 / (M * M), 0.001);
}

TEST(HnswSearchLayer, HandBuiltLineGraph) {
    // Points at x = 0..4 on a line, chained 0-1-2-3-4 on layer 0.
    HnswIndex idx(1, 2, 10, 8, 1);
    for (int i = 0; i < 5; ++i) {
        const float x = static_cast<float>(i);
        HnswTestAccess::allocate(idx, &x, 0);
    }
    HnswTestAccess::set(idx, 0, 0, {1});
    HnswTestAccess::set(idx, 1, 0, {0, 2});
    HnswTestAccess::set(idx, 2, 0, {1, 3});
    HnswTestAccess::set(idx, 3, 0, {2, 4});
    HnswTestAccess::set(idx, 4, 0, {3});

    const float q = 3.2f;
    const float d0 = 3.2f * 3.2f;
    const auto r = HnswTestAccess::search_layer(idx, &q, {{d0, 0}}, 2, 0);
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].second, 3u);
    EXPECT_EQ(r[1].second, 4u);
    EXPECT_LE(r[0].first, r[1].first);

    // ef = 1 is plain greedy descent.
    const auto g = HnswTestAccess::search_layer(idx, &q, {{d0, 0}}, 1, 0);
    ASSERT_EQ(g.size(), 1u);
    EXPECT_EQ(g[0].second, 3u);
}
