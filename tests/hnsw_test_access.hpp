#pragma once

#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "hnsw/hnsw_index.hpp"

namespace hnsw {

// Friend of HnswIndex: pokes private internals from tests.
struct HnswTestAccess {
    static uint32_t allocate(HnswIndex& i, const float* v, int level) {
        return i.allocate_node(v, level);
    }
    static std::span<const uint32_t> nbrs(const HnswIndex& i, uint32_t id, int layer) {
        return i.neighbors(id, layer);
    }
    static void set(HnswIndex& i, uint32_t id, int layer, std::vector<uint32_t> ns) {
        i.set_neighbors(id, layer, ns);
    }
    static const float* vec(const HnswIndex& i, uint32_t id) { return i.vector_at(id); }
    static size_t max_degree(const HnswIndex& i, int layer) { return i.max_degree(layer); }
    static int random_level(HnswIndex& i) { return i.random_level(); }
    static std::vector<Neighbor> search_layer(const HnswIndex& i, const float* q,
                                              std::vector<Neighbor> eps, size_t ef, int layer) {
        return i.search_layer(q, eps, ef, layer);
    }
    static int max_level(const HnswIndex& i) { return i.max_level_; }
    static int level_of(const HnswIndex& i, uint32_t id) { return i.levels_[id]; }

    // Structural invariants of a built graph.
    static void validate(const HnswIndex& i) {
        ASSERT_NE(i.entry_point_, HnswIndex::kInvalidId);
        EXPECT_EQ(i.levels_[i.entry_point_], i.max_level_);
        for (uint32_t id = 0; id < i.count_; ++id) {
            EXPECT_LE(i.levels_[id], i.max_level_);
            for (int l = 0; l <= i.levels_[id]; ++l) {
                const auto ns = i.neighbors(id, l);
                EXPECT_LE(ns.size(), i.max_degree(l)) << "node " << id << " layer " << l;
                for (size_t a = 0; a < ns.size(); ++a) {
                    EXPECT_LT(ns[a], i.count_);
                    EXPECT_NE(ns[a], id) << "self loop at " << id;
                    EXPECT_GE(i.levels_[ns[a]], l) << "neighbor not present on layer " << l;
                    for (size_t b = a + 1; b < ns.size(); ++b) EXPECT_NE(ns[a], ns[b]);
                }
            }
        }
    }
};

}  // namespace hnsw
