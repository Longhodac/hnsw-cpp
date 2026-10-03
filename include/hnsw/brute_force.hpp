#pragma once

#include <vector>

#include "hnsw/index.hpp"

namespace hnsw {

// Exact k-NN by scanning every vector. The correctness baseline for recall.
class BruteForceIndex final : public Index {
public:
    explicit BruteForceIndex(size_t dim) : dim_(dim) {}

    uint32_t add(const float* vec) override;
    std::vector<Neighbor> search(const float* query, size_t k, size_t ef_search) const override;

    size_t size() const override { return vectors_.size() / dim_; }
    size_t dim() const override { return dim_; }
    std::string_view name() const override { return "bruteforce"; }

private:
    size_t dim_;
    std::vector<float> vectors_;  // flat: id * dim_ + j
};

}  // namespace hnsw
