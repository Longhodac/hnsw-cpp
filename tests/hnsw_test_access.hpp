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
    static std::vector<uint32_t> nbrs(const HnswIndex& i, uint32_t id, int layer) {
        return i.neighbors(id, layer);
    }
    // Address of a node's link block, to check that storage never moves.
    static const void* link_ptr(const HnswIndex& i, uint32_t id, int layer) {
        return i.link_block(id, layer);
    }
    static void set(HnswIndex& i, uint32_t id, int layer, std::vector<uint32_t> ns) {
        i.set_neighbors(id, layer, ns);
    }
    static void add_link(HnswIndex& i, uint32_t node, uint32_t nbr, float d, int layer) {
        i.add_link(node, nbr, d, layer);
    }
    static const float* vec(const HnswIndex& i, uint32_t id) { return i.vector_at(id); }
    static size_t max_degree(const HnswIndex& i, int layer) { return i.max_degree(layer); }
    static int random_level(const HnswIndex& i, uint32_t id) { return i.random_level(id); }
    static std::vector<Neighbor> search_layer(const HnswIndex& i, const float* q,
                                              std::vector<Neighbor> eps, size_t ef, int layer) {
        return i.search_layer(q, eps, ef, layer);
    }
    static int max_level(const HnswIndex& i) { return i.load_entry().level; }
    static int level_of(const HnswIndex& i, uint32_t id) { return i.levels_[id]; }

    // Structural invariants of a built graph.
    static void validate(const HnswIndex& i) {
        const auto entry = i.load_entry();
        ASSERT_NE(entry.id, HnswIndex::kInvalidId);
        EXPECT_EQ(i.levels_[entry.id], entry.level);
        for (uint32_t id = 0; id < i.count_.load(); ++id) {
            EXPECT_LE(i.levels_[id], entry.level);
            for (int l = 0; l <= i.levels_[id]; ++l) {
                const auto ns = i.neighbors(id, l);
                EXPECT_LE(ns.size(), i.max_degree(l)) << "node " << id << " layer " << l;
                for (size_t a = 0; a < ns.size(); ++a) {
                    EXPECT_LT(ns[a], i.count_.load());
                    EXPECT_NE(ns[a], id) << "self loop at " << id;
                    EXPECT_GE(i.levels_[ns[a]], l) << "neighbor not present on layer " << l;
                    for (size_t b = a + 1; b < ns.size(); ++b) EXPECT_NE(ns[a], ns[b]);
                }
            }
        }
    }
};

}  // namespace hnsw
