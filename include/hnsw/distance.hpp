#pragma once

#include <cstddef>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define HNSW_HAS_NEON 1
#endif

namespace hnsw {

// Reference implementation. Adds in index order, so it is the exact baseline the
// vectorized version is tested against. Used directly on CPUs without NEON.
inline float l2_sqr_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

#if defined(HNSW_HAS_NEON)
// NEON version. Each register holds 4 floats. Four independent accumulators keep four
// fused multiply-adds in flight, which breaks the serial add chain of the scalar loop.
// The partial sums are combined once at the end, so the result can differ from
// l2_sqr_scalar in the last few digits. Loads are unaligned-safe.
inline float l2_sqr_neon(const float* a, const float* b, size_t dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d, d);
    }

    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

// Squared Euclidean distance. Kept inline in the header so index code can inline it
// into its inner loops.
inline float l2_sqr(const float* a, const float* b, size_t dim) {
#if defined(HNSW_HAS_NEON)
    return l2_sqr_neon(a, b, dim);
#else
    return l2_sqr_scalar(a, b, dim);
#endif
}

}  // namespace hnsw
