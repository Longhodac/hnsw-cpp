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

}  // namespace

HnswIndex::HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements,
                     uint64_t seed)
    : dim_(dim),
      M_(M),
      max_elements_(max_elements),
      rng_(seed),
      ef_construction_(ef_construction),
      mL_(M >= 2 ? 1.0 / std::log(static_cast<double>(M)) : 0.0) {
    if (dim == 0) throw std::invalid_argument("HnswIndex: dim must be > 0");
    if (M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
    if (max_elements >= kInvalidId) {
        throw std::invalid_argument("HnswIndex: max_elements must fit in uint32_t ids");
    }
    // Everything is sized once, so no later insert reallocates or moves data other threads read.
    vectors_ = std::make_unique_for_overwrite<float[]>(max_elements * dim);  // pages fault in lazily
    levels_ = std::make_unique<uint8_t[]>(max_elements);
    level0_links_.assign(max_elements * (1 + 2 * M), 0);  // counts start at 0 = no neighbors
    upper_links_.resize(max_elements);
}

uint32_t HnswIndex::allocate_node(const float* vec, int level) {
    if (level < 0 || level > std::numeric_limits<uint8_t>::max()) {
        throw std::invalid_argument("HnswIndex: level out of range");
    }
    size_t next = count_.load(std::memory_order_relaxed);
    do {
        if (next >= max_elements_) throw std::length_error("HnswIndex: max_elements reached");
    } while (!count_.compare_exchange_weak(next, next + 1, std::memory_order_relaxed));

    const auto id = static_cast<uint32_t>(next);
    std::copy(vec, vec + dim_, vectors_.get() + static_cast<size_t>(id) * dim_);
    levels_[id] = static_cast<uint8_t>(level);
    if (level > 0) {
        // One zeroed block (count + M slots) per layer 1..level.
        upper_links_[id] = std::make_unique<uint32_t[]>(static_cast<size_t>(level) * (1 + M_));
    }
    return id;
}

const uint32_t* HnswIndex::link_block(uint32_t id, int layer) const {
    assert(id < count_ && layer >= 0 && layer <= levels_[id]);
    if (layer == 0) return level0_links_.data() + static_cast<size_t>(id) * (1 + 2 * M_);
    return upper_links_[id].get() + static_cast<size_t>(layer - 1) * (1 + M_);
}

uint32_t* HnswIndex::link_block(uint32_t id, int layer) {
    return const_cast<uint32_t*>(std::as_const(*this).link_block(id, layer));
}

std::span<const uint32_t> HnswIndex::neighbors(uint32_t id, int layer) const {
    const uint32_t* block = link_block(id, layer);
    return {block + 1, block[0]};
}

void HnswIndex::set_neighbors(uint32_t id, int layer, std::span<const uint32_t> ns) {
    assert(ns.size() <= max_degree(layer));
    uint32_t* block = link_block(id, layer);
    block[0] = static_cast<uint32_t>(ns.size());
    std::copy(ns.begin(), ns.end(), block + 1);
}

int HnswIndex::random_level() {
    std::uniform_real_distribution<double> unif(0.0, 1.0);  // [0, 1)
    const double u = 1.0 - unif(rng_);                      // (0, 1], so log(u) is finite
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

        // Start loading every neighbor's visited tag and vector before using any of them,
        // so the memory reads overlap instead of running one after another.
        const std::span<const uint32_t> nbrs = neighbors(c.second, layer);
        for (const uint32_t e : nbrs) {
            __builtin_prefetch(&tags[e]);
            prefetch_bytes(vector_at(e), dim_ * sizeof(float));
        }

        for (const uint32_t e : nbrs) {
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
    for (bool improved = true; improved;) {
        improved = false;
        for (const uint32_t e : neighbors(cur.second, layer)) {
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

void HnswIndex::add_link(uint32_t node, uint32_t new_nbr, float dist, int layer) {
    uint32_t* block = link_block(node, layer);
    const size_t count = block[0];
    const size_t cap = max_degree(layer);
    if (count < cap) {
        block[1 + count] = new_nbr;
        block[0] = static_cast<uint32_t>(count + 1);
        return;
    }
    // Full: choose the best `cap` among current neighbors plus the newcomer.
    std::vector<Neighbor> cands;
    cands.reserve(count + 1);
    cands.emplace_back(dist, new_nbr);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = block[1 + i];
        cands.emplace_back(l2_sqr(vector_at(node), vector_at(id), dim_), id);
    }
    std::sort(cands.begin(), cands.end());
    const auto kept = select_neighbors(cands, cap);
    block[0] = static_cast<uint32_t>(kept.size());
    for (size_t i = 0; i < kept.size(); ++i) block[1 + i] = kept[i].second;
}

uint32_t HnswIndex::add(const float* vec) {
    const int level = random_level();
    const uint32_t id = allocate_node(vec, level);  // throws if full

    if (entry_point_ == kInvalidId) {  // first node: it is the whole graph
        entry_point_ = id;
        max_level_ = level;
        return id;
    }

    Neighbor ep{l2_sqr(vec, vector_at(entry_point_), dim_), entry_point_};

    // Layers above the new node's top: only navigate (ef = 1).
    for (int lc = max_level_; lc > level; --lc) ep = greedy_closest(vec, ep, lc);

    // Layers the node lives on: find candidates, link, and prune neighbors' lists.
    std::vector<Neighbor> entry_points{ep};
    for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
        std::vector<Neighbor> found = search_layer(vec, entry_points, ef_construction_, lc);
        const std::vector<Neighbor> chosen = select_neighbors(found, M_);

        uint32_t* block = link_block(id, lc);
        block[0] = static_cast<uint32_t>(chosen.size());
        for (size_t i = 0; i < chosen.size(); ++i) block[1 + i] = chosen[i].second;
        for (const Neighbor& n : chosen) add_link(n.second, id, n.first, lc);

        entry_points = std::move(found);  // next layer down starts from everything found here
    }

    if (level > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
    return id;
}

std::vector<Neighbor> HnswIndex::search(const float* query, size_t k, size_t ef_search) const {
    if (count_ == 0 || k == 0) return {};

    Neighbor ep{l2_sqr(query, vector_at(entry_point_), dim_), entry_point_};
    for (int lc = max_level_; lc > 0; --lc) ep = greedy_closest(query, ep, lc);

    const std::array<Neighbor, 1> eps{ep};
    std::vector<Neighbor> result = search_layer(query, eps, std::max(ef_search, k), 0);
    if (result.size() > k) result.resize(k);
    return result;
}

}  // namespace hnsw
