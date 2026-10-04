#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "hnsw/index.hpp"
#include "hnsw/index_file.hpp"
#include "hnsw/memory_region.hpp"

namespace hnsw {

// Hierarchical Navigable Small World graph index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" (arXiv:1603.09320).
//
// Thread safety: add() and search() can both be called from many threads at once, up to
// max_elements total. Searches take no locks. Inserts take one short per-node lock at a time
// while they rewrite a neighbor list, and one global lock only when a new node raises the top
// layer. A search that runs during inserts sees a valid graph, though maybe not the newest
// nodes. With several inserting threads the graph differs from run to run, and size() counts
// nodes whose insert has started, not only finished.
//
// Persistence: save() writes the index to one file and load() maps it back in place, so the
// load takes microseconds and pages come in from disk as searches touch them. A loaded index
// is read-only: add() throws. Only load files this program wrote. load() checks the header and
// section sizes but not every link, so a hand-damaged file can still hold out-of-range ids;
// LoadOptions::verify scans all of them.
class HnswIndex final : public Index {
public:
    struct LoadOptions {
        bool verify = false;  // also check every link, which reads the whole file
    };

    HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements, uint64_t seed);

    // Inserts one vector, returns its id.   (Algorithm 1: INSERT)
    uint32_t add(const float* vec) override;

    // k-NN query, sorted by ascending distance.   (Algorithm 5: K-NN-SEARCH)
    std::vector<Neighbor> search(const float* query, size_t k, size_t ef_search) const override;

    size_t size() const override { return count_.load(); }
    size_t dim() const override { return dim_; }
    std::string_view name() const override { return use_heuristic_ ? "hnsw" : "hnsw-simple"; }

    // Neighbor selection: true (default) = diversity heuristic (Algorithm 4),
    // false = plain M closest (Algorithm 3). Set before adding elements.
    void set_use_heuristic(bool on) { use_heuristic_ = on; }

    // Writes the index to `path` through a temporary file and a rename, so a crash cannot leave
    // a half-written file under that name. Nothing may call add() while this runs. Throws
    // std::runtime_error on an I/O failure.
    void save(const std::string& path) const;

    // Maps a saved index. The result is read-only. Throws std::runtime_error if the file is
    // missing, damaged, truncated or from an incompatible version.
    static std::unique_ptr<HnswIndex> load(const std::string& path);
    static std::unique_ptr<HnswIndex> load(const std::string& path, LoadOptions options);

    bool read_only() const { return read_only_; }

private:
    friend struct HnswTestAccess;  // lets unit tests exercise the storage layer

    static constexpr uint32_t kInvalidId = UINT32_MAX;
    static constexpr size_t kLockStripes = 4096;  // power of two

    struct LoadTag {};
    // Wraps a mapped file whose header has already passed index_file::validate.
    HnswIndex(LoadTag, MemoryRegion mapped, const index_file::Header& header);

    size_t dim_;
    size_t M_;
    size_t max_elements_;
    uint64_t seed_;  // levels are a hash of (seed_, id), so they do not depend on thread timing

    std::atomic<size_t> count_{0};  // ids handed out so far

    // ---- Graph storage layout (fixed capacity, nothing reallocates or moves) ----
    //
    // Each array lives in its own MemoryRegion (an anonymous mmap: zero-filled, and physical
    // memory is used only for pages that get touched). The raw pointers below point into them.
    //
    // vectors_       id * dim_ + j                       all vectors, flat
    // levels_[id]    top layer of node id (0 = layer 0 only)
    //
    // Layer 0 (every node): one fixed block per node, stride 1 + 2M slots:
    //     level0_[id * (1 + 2M)] = [count, n_0, n_1, ... n_{2M-1}]
    //
    // Layers >= 1 (only nodes with level > 0): upper_offset_[id] is where node id's blocks start
    // in upper_arena_, in slots (0 = the node has none). A node with level l owns l blocks of
    // stride 1 + M, back to back:
    //     upper_arena_[upper_offset_[id] + (layer - 1) * (1 + M)] = [count, n_0 ... n_{M-1}]
    //   Blocks come from a bump allocator over a fixed-size arena; slot 0 is never handed out.
    //   About 1 node in M has upper layers, so the arena is small.
    //
    // A block's first slot is the live neighbor count; slots after it are neighbor ids.
    // Slots are plain uint32_t read and written through std::atomic_ref, so the same bytes can
    // sit in a file mapping.
    //
    // Publication: a node's vector, level and upper offset are written before its id is stored
    // into any neighbor slot. Slots are stored with release and loaded with acquire, so a thread
    // that reads an id from a slot also sees everything written for that node.
    size_t ef_construction_;
    double mL_;  // level multiplier, 1 / ln(M)
    MemoryRegion vectors_region_, levels_region_, level0_region_, upper_offset_region_,
        upper_arena_region_;
    MemoryRegion file_region_;  // set instead of the five above when loaded from a file
    bool read_only_ = false;
    float* vectors_ = nullptr;
    uint8_t* levels_ = nullptr;
    uint32_t* level0_ = nullptr;
    uint32_t* upper_offset_ = nullptr;
    uint32_t* upper_arena_ = nullptr;
    size_t upper_arena_capacity_ = 0;                // slots
    std::atomic<size_t> upper_arena_used_{1};        // slots handed out; slot 0 stays unused

    // Entry node and its layer, packed into one word so a reader always sees a matching pair.
    // Written only while holding entry_mutex_.
    struct Entry {
        uint32_t id;
        int level;  // -1 while the graph is empty
    };
    static uint64_t pack(Entry e) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(e.level + 1)) << 32) | e.id;
    }
    static Entry unpack(uint64_t w) {
        return {static_cast<uint32_t>(w), static_cast<int>(w >> 32) - 1};
    }
    Entry load_entry() const { return unpack(entry_.load(std::memory_order_acquire)); }
    std::atomic<uint64_t> entry_{pack({kInvalidId, -1})};
    std::mutex entry_mutex_;

    // Striped per-node locks: node id uses link_locks_[id % kLockStripes]. A thread holds at
    // most one of them at a time, so they cannot deadlock.
    std::unique_ptr<std::mutex[]> link_locks_;
    std::mutex& lock_for(uint32_t id) { return link_locks_[id & (kLockStripes - 1)]; }

    bool use_heuristic_ = true;

    // Keep candidates rejected by the heuristic to fill up to M (paper's
    // keepPrunedConnections). Off, matching hnswlib's default.
    static constexpr bool kKeepPruned = false;

    // Max neighbors per node on a layer: 2M on layer 0, M above (Mmax0 / Mmax in the paper).
    size_t max_degree(int layer) const { return layer == 0 ? 2 * M_ : M_; }

    const float* vector_at(uint32_t id) const {
        return vectors_ + static_cast<size_t>(id) * dim_;
    }

    // Takes the next id. Throws std::length_error when the index is full.
    uint32_t reserve_id();

    // Writes the vector and level of a reserved id and allocates its upper-layer link blocks.
    // Layer 0 blocks are already zeroed. Does NOT touch the entry point. If the arena is full
    // it throws and the reserved id stays unused, so treat the index as full.
    void init_node(uint32_t id, const float* vec, int level);

    // reserve_id + init_node.
    uint32_t allocate_node(const float* vec, int level);

    // Copies the neighbor list of `id` on `layer` (requires layer <= levels_[id]) into `out`,
    // which must hold max_degree(layer) ids. Returns the count.
    size_t snapshot_neighbors(uint32_t id, int layer, uint32_t* out) const;

    // Same list as a vector. For tests and validation.
    std::vector<uint32_t> neighbors(uint32_t id, int layer) const;

    // Replaces the neighbor list (size must be <= max_degree(layer)). Not safe against
    // concurrent writers to the same node. For tests.
    void set_neighbors(uint32_t id, int layer, std::span<const uint32_t> ns);

    uint32_t* link_block(uint32_t id, int layer) const;

    // Hands out `slots` consecutive slots from the arena and returns the offset of the first.
    // Throws std::length_error when the arena is full. The arena is sized at about 8 times
    // the expected need, so this means a wildly unlucky level sequence.
    uint32_t alloc_upper(size_t slots);

    // Level l = floor(-ln(U) * mL), U in (0, 1] from a hash of (seed_, id); P(level >= l) = M^-l.
    int random_level(uint32_t id) const;

    // Algorithm 2. Best-first search on one layer from the given entry points; returns
    // the (up to) ef closest nodes found, sorted by ascending distance.
    std::vector<Neighbor> search_layer(const float* query, std::span<const Neighbor> entry_points,
                                       size_t ef, int layer) const;

    // search_layer with ef = 1 on one layer, as a plain greedy walk. Returns the closest node.
    Neighbor greedy_closest(const float* query, Neighbor entry, int layer) const;

    // Algorithms 3/4. `candidates` must be sorted ascending by distance to the base node.
    std::vector<Neighbor> select_neighbors(const std::vector<Neighbor>& candidates,
                                           size_t m) const;

    // Adds new_nbr to node's list on `layer` under node's lock; if full, re-selects the best
    // max_degree neighbors among the old ones plus new_nbr. `dist` = distance(node, new_nbr).
    void add_link(uint32_t node, uint32_t new_nbr, float dist, int layer);

    // add_link for a caller that already holds lock_for(node). Skips ids already in the list.
    void add_link_locked(uint32_t node, uint32_t new_nbr, float dist, int layer);

    // Throws std::runtime_error describing the first structural problem: an entry node that
    // disagrees with the levels, an upper offset outside the arena, a neighbor count above the
    // layer's limit, or a link to a missing node, to itself, or to a node not on that layer.
    // Reads every link. For loaded files.
    void verify_structure() const;

    // Fills a new node's own list on `layer` with `chosen`, merging with any links other
    // threads already added to it.
    void set_own_links(uint32_t id, const std::vector<Neighbor>& chosen, int layer);
};

}  // namespace hnsw
