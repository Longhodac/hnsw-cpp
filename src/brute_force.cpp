#include "hnsw/brute_force.hpp"

#include <algorithm>
#include <queue>

#include "hnsw/distance.hpp"

namespace hnsw {

uint32_t BruteForceIndex::add(const float* vec) {
    const auto id = static_cast<uint32_t>(size());
    vectors_.insert(vectors_.end(), vec, vec + dim_);
    return id;
}

std::vector<Neighbor> BruteForceIndex::search(const float* query, size_t k,
                                              size_t /*ef_search*/) const {
    // Max-heap of the k best so far; top() is the current worst. Neighbor ordering is
    // (distance, id), so ties resolve toward the lower id.
    std::priority_queue<Neighbor> heap;
    const size_t n = size();
    for (size_t i = 0; i < n; ++i) {
        const Neighbor cand{l2_sqr(query, vectors_.data() + i * dim_, dim_),
                            static_cast<uint32_t>(i)};
        if (heap.size() < k) {
            heap.push(cand);
        } else if (cand < heap.top()) {
            heap.pop();
            heap.push(cand);
        }
    }
    std::vector<Neighbor> out(heap.size());
    for (size_t i = out.size(); i-- > 0;) {
        out[i] = heap.top();
        heap.pop();
    }
    return out;
}

}  // namespace hnsw
