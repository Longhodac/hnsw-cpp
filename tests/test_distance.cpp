#include <gtest/gtest.h>

#include <random>
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

#if defined(HNSW_HAS_NEON)
// The vectorized sum adds in a different order than the scalar one, so compare within a
// relative tolerance. Covers every dim from 0 to 300: below one block, the 16-wide loop,
// the 4-wide loop, and the scalar tail.
TEST(DistanceNeon, MatchesScalarForEveryDim) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-100.f, 100.f);
    for (size_t dim = 0; dim <= 300; ++dim) {
        std::vector<float> a(dim), b(dim);
        for (auto& x : a) x = u(rng);
        for (auto& x : b) x = u(rng);
        const float ref = hnsw::l2_sqr_scalar(a.data(), b.data(), dim);
        const float got = hnsw::l2_sqr_neon(a.data(), b.data(), dim);
        EXPECT_NEAR(got, ref, 1e-5f * ref + 1e-6f) << "dim=" << dim;
    }
}

TEST(DistanceNeon, KnownValuesWithTail) {
    // dim 21 = one 16-wide block + one 4-wide block + 1 tail element.
    std::vector<float> a(21, 1.f), b(21, 3.f);
    EXPECT_FLOAT_EQ(hnsw::l2_sqr_neon(a.data(), b.data(), 21), 21.f * 4.f);
}

TEST(DistanceNeon, UnalignedPointers) {
    std::vector<float> a(200), b(200);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 0.5f;
    }
    for (size_t offset = 0; offset < 4; ++offset) {
        const size_t dim = 128;
        const float ref = hnsw::l2_sqr_scalar(a.data() + offset, b.data() + offset, dim);
        const float got = hnsw::l2_sqr_neon(a.data() + offset, b.data() + offset, dim);
        EXPECT_NEAR(got, ref, 1e-5f * ref) << "offset=" << offset;
    }
}

TEST(DistanceNeon, ZeroForIdentical) {
    const std::vector<float> a(131, 2.5f);
    EXPECT_EQ(hnsw::l2_sqr_neon(a.data(), a.data(), a.size()), 0.f);
}
#endif
