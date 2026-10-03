#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace hnsw {

// (squared L2 distance, node id). std::pair orders by distance, then id, so
// sorting or heap-ing Neighbors gives deterministic tie-breaking.
using Neighbor = std::pair<float, uint32_t>;

// Common interface so tools/eval can drive any index (brute force, HNSW, ...).
class Index {
public:
    virtual ~Index() = default;

    // Appends one vector of dim() floats; returns its id (ids are dense, 0..size()-1).
    virtual uint32_t add(const float* vec) = 0;

    // Returns up to k nearest neighbors, sorted by ascending distance.
    // ef_search is the HNSW beam width; exact indexes ignore it.
    virtual std::vector<Neighbor> search(const float* query, size_t k, size_t ef_search) const = 0;

    virtual size_t size() const = 0;
    virtual size_t dim() const = 0;
    virtual std::string_view name() const = 0;
};

}  // namespace hnsw
