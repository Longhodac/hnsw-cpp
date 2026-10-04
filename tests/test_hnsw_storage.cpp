#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>
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
    for (uint32_t id = 0; id < 1000; ++id) {
        EXPECT_EQ(HnswTestAccess::random_level(a, id), HnswTestAccess::random_level(b, id));
    }
}

TEST(HnswLevels, DependOnlyOnSeedAndId) {
    HnswIndex a(2, 16, 10, 8, 7), c(2, 16, 10, 8, 8);
    int differing = 0;
    for (uint32_t id = 0; id < 5000; ++id) {
        // Same answer on a repeat call: no hidden generator state.
        EXPECT_EQ(HnswTestAccess::random_level(a, id), HnswTestAccess::random_level(a, id));
        differing += HnswTestAccess::random_level(a, id) != HnswTestAccess::random_level(c, id);
    }
    EXPECT_GT(differing, 0);  // a different seed gives a different level sequence
}

TEST(HnswLevels, GeometricDistribution) {
    const size_t M = 16;
    HnswIndex idx(2, M, 10, 8, 123);
    const int n = 200000;
    std::vector<int> hist(32, 0);
    for (int i = 0; i < n; ++i) {
        ++hist[std::min(HnswTestAccess::random_level(idx, static_cast<uint32_t>(i)), 31)];
    }

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

TEST(HnswStorage, StorageNeverMovesOnInsert) {
    HnswIndex idx(2, 4, 10, 64, 1);
    const float v[] = {1, 2};
    const auto first = HnswTestAccess::allocate(idx, v, 2);
    const float* vec_before = HnswTestAccess::vec(idx, first);
    const void* links_before[3];
    for (int l = 0; l <= 2; ++l) links_before[l] = HnswTestAccess::link_ptr(idx, first, l);

    for (int i = 0; i < 40; ++i) HnswTestAccess::allocate(idx, v, i % 4);  // incl. upper layers

    EXPECT_EQ(HnswTestAccess::vec(idx, first), vec_before);
    for (int l = 0; l <= 2; ++l) {
        EXPECT_EQ(HnswTestAccess::link_ptr(idx, first, l), links_before[l]) << "layer " << l;
    }
}

TEST(HnswStorage, ConcurrentAllocateGivesUniqueDenseIds) {
    constexpr size_t kThreads = 4, kPerThread = 250;
    HnswIndex idx(2, 4, 10, kThreads * kPerThread, 1);
    std::vector<std::vector<uint32_t>> ids(kThreads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            const float v[] = {static_cast<float>(t), 0};
            for (size_t i = 0; i < kPerThread; ++i) {
                ids[t].push_back(HnswTestAccess::allocate(idx, v, static_cast<int>(i % 3)));
            }
        });
    }
    for (auto& th : threads) th.join();

    std::vector<uint32_t> all;
    for (const auto& v : ids) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    ASSERT_EQ(all.size(), kThreads * kPerThread);
    for (size_t i = 0; i < all.size(); ++i) EXPECT_EQ(all[i], i);
    EXPECT_EQ(idx.size(), kThreads * kPerThread);
    // Each thread's vectors landed intact in its own slots.
    for (size_t t = 0; t < kThreads; ++t) {
        for (const uint32_t id : ids[t]) EXPECT_EQ(HnswTestAccess::vec(idx, id)[0], static_cast<float>(t));
    }
}

TEST(HnswStorage, ConcurrentAllocateStopsAtCapacity) {
    constexpr size_t kThreads = 4, kCap = 100;
    HnswIndex idx(1, 4, 10, kCap, 1);
    std::atomic<size_t> ok{0}, full{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            const float v[] = {0};
            for (int i = 0; i < 100; ++i) {
                try {
                    HnswTestAccess::allocate(idx, v, 0);
                    ++ok;
                } catch (const std::length_error&) {
                    ++full;
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(ok.load(), kCap);
    EXPECT_EQ(full.load(), kThreads * 100 - kCap);
    EXPECT_EQ(idx.size(), kCap);
}

TEST(HnswStorage, AddLinkRefusesSelfLinksAndDuplicates) {
    HnswIndex idx(1, 4, 10, 8, 1);
    const float v[] = {0};
    HnswTestAccess::allocate(idx, v, 0);
    HnswTestAccess::allocate(idx, v, 0);
    HnswTestAccess::add_link(idx, 0, 0, 0.f, 0);  // self link: ignored
    EXPECT_TRUE(HnswTestAccess::nbrs(idx, 0, 0).empty());
    HnswTestAccess::add_link(idx, 0, 1, 1.f, 0);
    HnswTestAccess::add_link(idx, 0, 1, 1.f, 0);  // duplicate: ignored
    EXPECT_EQ(HnswTestAccess::nbrs(idx, 0, 0), std::vector<uint32_t>{1});
}
