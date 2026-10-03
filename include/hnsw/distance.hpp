#pragma once

#include <cstddef>

namespace hnsw {

// Squared Euclidean distance. Scalar on purpose: SIMD comes in Phase 4.
// Kept inline in the header so index code can inline it into its inner loops.
inline float l2_sqr(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace hnsw
