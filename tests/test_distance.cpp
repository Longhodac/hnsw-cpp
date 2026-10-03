#include <gtest/gtest.h>

#include <vector>

#include "hnsw/distance.hpp"

TEST(Distance, KnownValue) {
    const float a[] = {1, 2, 3};
    const float b[] = {4, 6, 3};
    EXPECT_FLOAT_EQ(hnsw::l2_sqr(a, b, 3), 25.f);  // 9 + 16 + 0
}

TEST(Distance, ZeroForIdentical) {
    const std::vector<float> a(128, 1.5f);
    EXPECT_EQ(hnsw::l2_sqr(a.data(), a.data(), a.size()), 0.f);
}

TEST(Distance, Symmetric) {
    std::vector<float> a(37), b(37);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(37 - i);
    }
    EXPECT_EQ(hnsw::l2_sqr(a.data(), b.data(), 37), hnsw::l2_sqr(b.data(), a.data(), 37));
}

TEST(Distance, ZeroDim) {
    const float a[] = {1};
    EXPECT_EQ(hnsw::l2_sqr(a, a, 0), 0.f);
}
