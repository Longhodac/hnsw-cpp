#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hnsw {

// A set of `count` fixed-dimension vectors stored contiguously (row-major).
template <typename T>
struct VecSet {
    size_t dim = 0;
    size_t count = 0;
    std::vector<T> data;  // count * dim elements

    const T* row(size_t i) const { return data.data() + i * dim; }
};

using FvecsData = VecSet<float>;
using IvecsData = VecSet<int32_t>;

// Load TEXMEX .fvecs / .ivecs files: each record is [int32 dim][dim x 4-byte value],
// little-endian. Throws std::runtime_error with a descriptive message if the file
// is missing, empty, truncated, or has inconsistent/invalid dimensions.
FvecsData load_fvecs(const std::string& path);
IvecsData load_ivecs(const std::string& path);

}  // namespace hnsw
