#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "hnsw/index.hpp"

namespace hnsw {

// Hierarchical Navigable Small World graph index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" (arXiv:1603.09320).
//
// Single-threaded: search() mutates internal scratch state (visited tags), so even
// concurrent searches are not yet safe (Phase 5).
class HnswIndex final : public Index {
public:
    HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements, uint64_t seed);

    // Inserts one vector, returns its id.   (Algorithm 1: INSERT)
    uint32_t add(const float* vec) override;

    // k-NN query, sorted by ascending distance.   (Algorithm 5: K-NN-SEARCH)
    std::vector<Neighbor> search(const float* query, size_t k, size_t ef_search) const override;

    size_t size() const override { return count_; }
    size_t dim() const override { return dim_; }
    std::string_view name() const override { return use_heuristic_ ? "hnsw" : "hnsw-simple"; }

    // Neighbor selection: true (default) = diversity heuristic (Algorithm 4),
    // false = plain M closest (Algorithm 3). Set before adding elements.
    void set_use_heuristic(bool on) { use_heuristic_ = on; }

private:
    friend struct HnswTestAccess;  // lets unit tests exercise the storage layer

    static constexpr uint32_t kInvalidId = UINT32_MAX;

    size_t dim_;
    size_t M_;
    size_t max_elements_;
    std::mt19937_64 rng_;  // for random level assignment

    size_t count_ = 0;
    std::vector<float> vectors_;  // flat: id * dim_ + j (reserved for max_elements_ * dim_)

    // ---- Graph storage layout (fixed capacity, no per-node heap allocations) ----
    //
    // vectors_       id * dim_ + j                       all vectors, flat
    // levels_[id]    top layer of node id (0 = layer 0 only)
    //
    // Layer 0 (every node): one fixed block per node, stride 1 + 2M uint32s:
    //     level0_links_[id * (1 + 2M)] = [count, n_0, n_1, ... n_{2M-1}]
    //   Allocated up front for max_elements_, so it never reallocates.
    //
    // Layers >= 1 (only nodes with level > 0): a node with level l owns l blocks of
    // stride 1 + M, stored contiguously and appended to upper_links_ on insertion:
    //     upper_links_[upper_offset_[id] + (layer - 1) * (1 + M)] = [count, n_0 ... n_{M-1}]
    //   upper_offset_[id] is only meaningful when levels_[id] > 0.
    //
    // A block's first slot is the live neighbor count; slots after it are neighbor ids.
    size_t ef_construction_;
    double mL_;                          // level multiplier, 1 / ln(M)
    std::vector<uint8_t> levels_;
    std::vector<uint32_t> level0_links_;
    std::vector<size_t> upper_offset_;
    std::vector<uint32_t> upper_links_;
    uint32_t entry_point_ = kInvalidId;  // id of the top-layer entry node
    int max_level_ = -1;                 // top layer currently in the graph (-1 = empty)
    bool use_heuristic_ = true;

    // Visited-set scratch for search_layer: visited_[id] == epoch_ means "seen this call".
    // Bumping epoch_ resets the whole set in O(1).
    mutable std::vector<uint32_t> visited_;
    mutable uint32_t epoch_ = 0;

    // Keep candidates rejected by the heuristic to fill up to M (paper's
    // keepPrunedConnections). Off, matching hnswlib's default.
    static constexpr bool kKeepPruned = false;

    // Max neighbors per node on a layer: 2M on layer 0, M above (Mmax0 / Mmax in the paper).
    size_t max_degree(int layer) const { return layer == 0 ? 2 * M_ : M_; }

    const float* vector_at(uint32_t id) const { return vectors_.data() + id * dim_; }

    // Reserves storage for a new node: copies the vector, records its level, and
    // zero-initializes its link blocks on layers 0..level. Returns the new id.
    // Throws std::length_error when the index is full. Does NOT touch the entry point.
    uint32_t allocate_node(const float* vec, int level);

    // Neighbor list of `id` on `layer` (requires layer <= levels_[id]).
    std::span<const uint32_t> neighbors(uint32_t id, int layer) const;

    // Replaces the neighbor list (size must be <= max_degree(layer)).
    void set_neighbors(uint32_t id, int layer, std::span<const uint32_t> ns);

    uint32_t* link_block(uint32_t id, int layer);
    const uint32_t* link_block(uint32_t id, int layer) const;

    // Level l = floor(-ln(U) * mL), U ~ uniform(0, 1]; P(level >= l) = M^-l.
    int random_level();

    void next_epoch() const;

    // Algorithm 2. Best-first search on one layer from the given entry points; returns
    // the (up to) ef closest nodes found, sorted by ascending distance.
    std::vector<Neighbor> search_layer(const float* query, std::span<const Neighbor> entry_points,
                                       size_t ef, int layer) const;

    // search_layer with ef = 1 on one layer, as a plain greedy walk. Returns the closest node.
    Neighbor greedy_closest(const float* query, Neighbor entry, int layer) const;

    // Algorithms 3/4. `candidates` must be sorted ascending by distance to the base node.
    std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& candidates,
                                           size_t m) const;

    // Adds new_nbr to node's list on `layer`; if full, re-selects the best max_degree
    // neighbors among the old ones plus new_nbr. `dist` = distance(node, new_nbr).
    void add_link(uint32_t node, uint32_t new_nbr, float dist, int layer);
};

}  // namespace hnsw
