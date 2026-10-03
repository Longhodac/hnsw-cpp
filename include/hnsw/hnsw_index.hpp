#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "hnsw/index.hpp"

namespace hnsw {

// Hierarchical Navigable Small World graph index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" (arXiv:1603.09320).
//
// INTERFACE ONLY: method bodies in src/hnsw_index.cpp are stubs that throw.
class HnswIndex final : public Index {
public:
    HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements, uint64_t seed);

    // Inserts one vector, returns its id.   (Algorithm 1: INSERT)
    uint32_t add(const float* vec) override;

    // k-NN query, sorted by ascending distance.   (Algorithm 5: K-NN-SEARCH)
    std::vector<Neighbor> search(const float* query, size_t k, size_t ef_search) const override;

    size_t size() const override { return count_; }
    size_t dim() const override { return dim_; }
    std::string_view name() const override { return "hnsw"; }

private:
    size_t dim_;
    [[maybe_unused]] size_t M_;  // remove attribute once used
    [[maybe_unused]] size_t ef_construction_;
    [[maybe_unused]] size_t max_elements_;
    std::mt19937_64 rng_;  // for random level assignment

    size_t count_ = 0;
    std::vector<float> vectors_;  // flat: id * dim_ + j (reserve max_elements_ * dim_)

    // TODO: add your graph storage (per-node level, per-layer neighbor lists, entry
    // point, max level, level multiplier mL = 1/ln(M)) and any helper methods
    // (search_layer, select_neighbors) here.
};

}  // namespace hnsw
