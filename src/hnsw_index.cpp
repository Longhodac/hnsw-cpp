#include "hnsw/hnsw_index.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

#include "hnsw/distance.hpp"

namespace hnsw {

namespace {

constexpr size_t kCacheLine = 128;  // Apple Silicon (hw.cachelinesize)

// Asks the CPU to start loading [p, p + bytes) into cache. Never changes results.
inline void prefetch_bytes(const void* p, size_t bytes) {
    const char* c = static_cast<const char*>(p);
    for (size_t off = 0; off < bytes; off += kCacheLine) __builtin_prefetch(c + off);
    __builtin_prefetch(c + bytes - 1);  // a vector that straddles one more line
}

// Visited-set scratch for search_layer, one per thread. tags[id] == epoch means "seen in this
// call". Bumping epoch resets the whole set in O(1). The tags are tied to one index; a thread
// that switches to another index (or an index of another size) pays one O(max_elements) reset.
// A new index reusing an old address is safe: epoch only grows, so stale tags never match.
struct VisitedScratch {
    std::vector<uint32_t> tags;
    uint32_t epoch = 0;
    const void* owner = nullptr;

    void begin(const void* index, size_t n) {
        if (owner != index || tags.size() != n) {
            tags.assign(n, 0);
            epoch = 0;
            owner = index;
        }
        if (++epoch == 0) {  // wrapped: stale tags could alias the new epoch
            std::fill(tags.begin(), tags.end(), 0u);
            epoch = 1;
        }
    }
};

thread_local VisitedScratch tls_visited;

// Neighbor-list slots are plain uint32_t, accessed through atomic_ref so searches can read a
// list while an insert rewrites it, and so the same bytes can later sit in a file mapping.
inline uint32_t load_acquire(const uint32_t* p) {
    return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(p)).load(std::memory_order_acquire);
}
inline uint32_t load_relaxed(const uint32_t* p) {
    return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(p)).load(std::memory_order_relaxed);
}
inline void store_release(uint32_t* p, uint32_t v) {
    std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release);
}

// Per-thread buffer that holds one neighbor-list snapshot. search_layer and greedy_closest
// never run inside each other, so they can share it.
uint32_t* nbr_buffer(size_t n) {
    thread_local std::vector<uint32_t> buf;
    if (buf.size() < n) buf.resize(n);
    return buf.data();
}

}  // namespace

HnswIndex::HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements,
                     uint64_t seed)
    : dim_(dim),
      M_(M),
      max_elements_(max_elements),
      seed_(seed),
      ef_construction_(ef_construction),
      mL_(M >= 2 ? 1.0 / std::log(static_cast<double>(M)) : 0.0) {
    if (dim == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
    if (M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
    if (max_elements >= kInvalidId) {
        throw std::invalid_argument("HnswIndex: max_elements must fit in uint32_t ids");
    }
    // Everything is sized once, so no later insert reallocates or moves data other threads read.
    // Anonymous regions are zero-filled and only use memory for pages that get touched.
    vectors_region_ = MemoryRegion::anonymous(max_elements * dim * sizeof(float));
    levels_region_ = MemoryRegion::anonymous(max_elements);
    level0_region_ = MemoryRegion::anonymous(max_elements * (1 + 2 * M) * sizeof(uint32_t));
    upper_offset_region_ = MemoryRegion::anonymous(max_elements * sizeof(uint32_t));
    vectors_ = static_cast<float*>(vectors_region_.data());
    levels_ = static_cast<uint8_t*>(levels_region_.data());
    level0_ = static_cast<uint32_t*>(level0_region_.data());
    upper_offset_ = static_cast<uint32_t*>(upper_offset_region_.data());

    // Each node has expected (1 + M) / (M - 1) upper slots. Reserve 8 times that, at least a
    // million slots. Untouched pages cost nothing, so a generous arena is free.
    const size_t expected = max_elements * (1 + M) / (M - 1) + 1;
    upper_arena_capacity_ = std::min<size_t>(std::max<size_t>(8 * expected, size_t{1} << 20),
                                             std::numeric_limits<uint32_t>::max());
    upper_arena_region_ = MemoryRegion::anonymous(upper_arena_capacity_ * sizeof(uint32_t));
    upper_arena_ = static_cast<uint32_t*>(upper_arena_region_.data());
    link_locks_ = std::make_unique<std::mutex[]>(kLockStripes);
}

uint32_t HnswIndex::reserve_id() {
    size_t next = count_.load(std::memory_order_relaxed);
    do {
        if (next >= max_elements_) throw std::length_error("HnswIndex: max_elements reached");
    } while (!count_.compare_exchange_weak(next, next + 1, std::memory_order_relaxed));
    return static_cast<uint32_t>(next);
}

uint32_t HnswIndex::alloc_upper(size_t slots) {
    const size_t offset = upper_arena_used_.fetch_add(slots, std::memory_order_relaxed);
    if (offset + slots > upper_arena_capacity_) {
        throw std::length_error("HnswIndex: upper-layer arena full");
    }
    return static_cast<uint32_t>(offset);
}

void HnswIndex::init_node(uint32_t id, const float* vec, int level) {
    // One zeroed block (count + M slots) per layer 1..level. Allocate first so a full arena
    // leaves the node untouched.
    const uint32_t offset = level > 0 ? alloc_upper(static_cast<size_t>(level) * (1 + M_)) : 0;
    std::copy(vec, vec + dim_, vectors_ + static_cast<size_t>(id) * dim_);
    levels_[id] = static_cast<uint8_t>(level);
    upper_offset_[id] = offset;
}

uint32_t HnswIndex::allocate_node(const float* vec, int level) {
    if (level < 0 || level > std::numeric_limits<uint8_t>::max()) {
        throw std::invalid_argument("HnswIndex: level out of range");
    }
    const uint32_t id = reserve_id();  // throws if full
    init_node(id, vec, level);
    return id;
}

uint32_t* HnswIndex::link_block(uint32_t id, int layer) const {
    assert(id < count_ && layer >= 0 && layer <= levels_[id]);
    if (layer == 0) return level0_ + static_cast<size_t>(id) * (1 + 2 * M_);
    assert(upper_offset_[id] != 0);
    return upper_arena_ + upper_offset_[id] + static_cast<size_t>(layer - 1) * (1 + M_);
}

size_t HnswIndex::snapshot_neighbors(uint32_t id, int layer, uint32_t* out) const {
    const uint32_t* block = link_block(id, layer);
    const size_t count = std::min<size_t>(load_acquire(&block[0]), max_degree(layer));
    for (size_t i = 0; i < count; ++i) out[i] = load_acquire(&block[1 + i]);
    return count;
}

std::vector<uint32_t> HnswIndex::neighbors(uint32_t id, int layer) const {
    std::vector<uint32_t> out(max_degree(layer));
    out.resize(snapshot_neighbors(id, layer, out.data()));
    return out;
}

void HnswIndex::set_neighbors(uint32_t id, int layer, std::span<const uint32_t> ns) {
    assert(ns.size() <= max_degree(layer));
    uint32_t* block = link_block(id, layer);
    for (size_t i = 0; i < ns.size(); ++i) store_release(&block[1 + i], ns[i]);
    store_release(&block[0], static_cast<uint32_t>(ns.size()));
}

int HnswIndex::random_level(uint32_t id) const {
    // splitmix64 finalizer over (seed, id): a well-mixed 64-bit value that depends only on
    // the id, so the level is the same whichever thread inserts the node.
    uint64_t z = seed_ + (static_cast<uint64_t>(id) + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = (static_cast<double>(z >> 11) + 1.0) * 0x1.0p-53;  // (0, 1], so log(u) is finite
    const double level = std::floor(-std::log(u) * mL_);
    return static_cast<int>(std::min(level, 255.0));
}

std::vector<Neighbor> HnswIndex::search_layer(const float* query,
                                              std::span<const Neighbor> entry_points, size_t ef,
                                              int layer) const {
    VisitedScratch& vis = tls_visited;
    vis.begin(this, max_elements_);
    uint32_t* const tags = vis.tags.data();
    const uint32_t epoch = vis.epoch;
    uint32_t* const nbrs = nbr_buffer(max_degree(layer));
    // candidates: min-heap, nearest first to expand. results: max-heap of the best ef,
    // furthest on top so it can be evicted.
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<>> candidates;
    std::priority_queue<Neighbor> results;

    for (const Neighbor& ep : entry_points) {
        if (tags[ep.second] == epoch) continue;
        tags[ep.second] = epoch;
        candidates.push(ep);
        results.push(ep);
        if (results.size() > ef) results.pop();
    }

    while (!candidates.empty()) {
        const Neighbor c = candidates.top();
        // Every remaining candidate is at least this far; once the result set is full
        // and its worst member is closer, nothing can improve it.
        if (results.size() >= ef && c.first > results.top().first) break;
        candidates.pop();

        // Copy the list once so both passes below see the same ids even if an insert is
        // rewriting it. Then start loading every neighbor's visited tag and vector before
        // using any of them, so the memory reads overlap instead of running one after another.
        const size_t n = snapshot_neighbors(c.second, layer, nbrs);
        for (size_t i = 0; i < n; ++i) {
            __builtin_prefetch(&tags[nbrs[i]]);
            prefetch_bytes(vector_at(nbrs[i]), dim_ * sizeof(float));
        }

        for (size_t i = 0; i < n; ++i) {
            const uint32_t e = nbrs[i];
            if (tags[e] == epoch) continue;
            tags[e] = epoch;
            const float d = l2_sqr(query, vector_at(e), dim_);
            if (results.size() < ef || d < results.top().first) {
                candidates.emplace(d, e);
                results.emplace(d, e);
                if (results.size() > ef) results.pop();
            }
        }
    }

    std::vector<Neighbor> out(results.size());
    for (size_t i = out.size(); i-- > 0;) {
        out[i] = results.top();
        results.pop();
    }
    return out;
}

Neighbor HnswIndex::greedy_closest(const float* query, Neighbor cur, int layer) const {
    uint32_t* const nbrs = nbr_buffer(max_degree(layer));
    for (bool improved = true; improved;) {
        improved = false;
        const size_t n = snapshot_neighbors(cur.second, layer, nbrs);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t e = nbrs[i];
            const float d = l2_sqr(query, vector_at(e), dim_);
            if (d < cur.first) {
                cur = {d, e};
                improved = true;
            }
        }
    }
    return cur;
}

std::vector<Neighbor> HnswIndex::select_neighbors(const std::vector<Neighbor>& candidates,
                                                  size_t m) const {
    if (candidates.size() <= m) return candidates;
    if (!use_heuristic_) return {candidates.begin(), candidates.begin() + static_cast<long>(m)};

    // Algorithm 4: accept e only if it is closer to the base node than to every
    // already-accepted neighbor; otherwise an accepted neighbor already "covers" it.
    std::vector<Neighbor> selected;
    std::vector<Neighbor> pruned;
    selected.reserve(m);
    for (const Neighbor& e : candidates) {
        if (selected.size() >= m) break;
        bool diverse = true;
        for (const Neighbor& r : selected) {
            if (l2_sqr(vector_at(e.second), vector_at(r.second), dim_) < e.first) {
                diverse = false;
                break;
            }
        }
        if (diverse) {
            selected.push_back(e);
        } else if (kKeepPruned) {
            pruned.push_back(e);
        }
    }
    for (size_t i = 0; kKeepPruned && selected.size() < m && i < pruned.size(); ++i) {
        selected.push_back(pruned[i]);
    }
    return selected;
}

void HnswIndex::add_link_locked(uint32_t node, uint32_t new_nbr, float dist, int layer) {
    if (new_nbr == node) return;  // never link a node to itself
    uint32_t* block = link_block(node, layer);
    // The lock makes this thread the only writer, so it can read its own list relaxed.
    const size_t count = load_relaxed(&block[0]);
    const size_t cap = max_degree(layer);
    for (size_t i = 0; i < count; ++i) {
        if (load_relaxed(&block[1 + i]) == new_nbr) return;  // already linked
    }
    if (count < cap) {
        store_release(&block[1 + count], new_nbr);
        store_release(&block[0], static_cast<uint32_t>(count + 1));
        return;
    }
    // Full: choose the best `cap` among current neighbors plus the newcomer.
    std::vector<Neighbor> cands;
    cands.reserve(count + 1);
    cands.emplace_back(dist, new_nbr);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = load_relaxed(&block[1 + i]);
        cands.emplace_back(l2_sqr(vector_at(node), vector_at(id), dim_), id);
    }
    std::sort(cands.begin(), cands.end());
    const auto kept = select_neighbors(cands, cap);
    // Readers may see a mix of old and new ids while this runs. Every slot always holds a
    // valid node id on this layer, so a mixed read only costs a duplicate or a skipped link.
    for (size_t i = 0; i < kept.size(); ++i) store_release(&block[1 + i], kept[i].second);
    store_release(&block[0], static_cast<uint32_t>(kept.size()));
}

void HnswIndex::add_link(uint32_t node, uint32_t new_nbr, float dist, int layer) {
    std::lock_guard<std::mutex> lock(lock_for(node));
    add_link_locked(node, new_nbr, dist, layer);
}

void HnswIndex::set_own_links(uint32_t id, const std::vector<Neighbor>& chosen, int layer) {
    // Another thread can already be adding links to this node on `layer` (it may reach the
    // node as an entry point from the layer above), so merge instead of overwriting.
    std::lock_guard<std::mutex> lock(lock_for(id));
    for (const Neighbor& n : chosen) add_link_locked(id, n.second, n.first, layer);
}

uint32_t HnswIndex::add(const float* vec) {
    if (read_only_) throw std::logic_error("HnswIndex: a loaded index is read-only");
    const uint32_t id = reserve_id();  // throws if full
    const int level = random_level(id);
    init_node(id, vec, level);

    // A node that raises the top layer holds entry_mutex_ for the whole insert, so only one
    // such insert runs at a time. The top layer never decreases, so a node at or below the
    // current top needs no lock.
    std::unique_lock<std::mutex> top_lock(entry_mutex_, std::defer_lock);
    if (level > load_entry().level) top_lock.lock();
    const Entry entry = load_entry();

    if (entry.id == kInvalidId) {  // first node: it is the whole graph
        entry_.store(pack({id, level}), std::memory_order_release);
        return id;
    }

    Neighbor ep{l2_sqr(vec, vector_at(entry.id), dim_), entry.id};

    // Layers above the new node's top: only navigate (ef = 1).
    for (int lc = entry.level; lc > level; --lc) ep = greedy_closest(vec, ep, lc);

    // Layers the node lives on: find candidates, link, and prune neighbors' lists.
    std::vector<Neighbor> entry_points{ep};
    for (int lc = std::min(level, entry.level); lc >= 0; --lc) {
        std::vector<Neighbor> found = search_layer(vec, entry_points, ef_construction_, lc);
        // Another thread may have linked a newer node to this one on layer lc already, which
        // lets this search reach the node itself at distance 0. Never pick a node as its own
        // neighbor.
        std::erase_if(found, [id](const Neighbor& n) { return n.second == id; });
        const std::vector<Neighbor> chosen = select_neighbors(found, M_);

        set_own_links(id, chosen, lc);
        for (const Neighbor& n : chosen) add_link(n.second, id, n.first, lc);

        entry_points = std::move(found);  // next layer down starts from everything found here
    }

    if (level > entry.level) entry_.store(pack({id, level}), std::memory_order_release);
    return id;
}

std::vector<Neighbor> HnswIndex::search(const float* query, size_t k, size_t ef_search) const {
    if (k == 0) return {};
    const Entry entry = load_entry();
    if (entry.id == kInvalidId) return {};  // empty, or the first insert has not finished

    Neighbor ep{l2_sqr(query, vector_at(entry.id), dim_), entry.id};
    for (int lc = entry.level; lc > 0; --lc) ep = greedy_closest(query, ep, lc);

    const std::array<Neighbor, 1> eps{ep};
    std::vector<Neighbor> result = search_layer(query, eps, std::max(ef_search, k), 0);
    if (result.size() > k) result.resize(k);
    return result;
}

}  // namespace hnsw
