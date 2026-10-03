#include "hnsw/hnsw_index.hpp"

#include <stdexcept>

namespace hnsw {

HnswIndex::HnswIndex(size_t dim, size_t M, size_t ef_construction, size_t max_elements,
                     uint64_t seed)
    : dim_(dim),
      M_(M),
      ef_construction_(ef_construction),
      max_elements_(max_elements),
      rng_(seed) {
    // TODO: reserve vectors_ (max_elements * dim) and any per-node graph storage.
}

uint32_t HnswIndex::add(const float* /*vec*/) {
    // TODO (Algorithm 1, INSERT): draw the node's level l = floor(-ln(unif(0,1)) * mL),
    //   then from the entry point greedily descend layers above l (ef = 1), and for each
    //   layer from min(l, top) down to 0: run SEARCH-LAYER with ef_construction, pick
    //   neighbors, connect bidirectionally, and shrink any neighbor list exceeding Mmax.
    //   Update the entry point if l > current top layer.
    // TODO (Algorithm 2, SEARCH-LAYER): best-first search with a candidate min-heap and a
    //   result max-heap of size ef, plus a visited set.
    // TODO (Algorithm 4, SELECT-NEIGHBORS-HEURISTIC): keep candidate e only if it is closer
    //   to the query than to every neighbor already selected (diversity heuristic).
    throw std::logic_error("HnswIndex::add: not implemented");
}

std::vector<Neighbor> HnswIndex::search(const float* /*query*/, size_t /*k*/,
                                        size_t /*ef_search*/) const {
    // TODO (Algorithm 5, K-NN-SEARCH): greedy descent from the entry point through layers
    //   top..1 with ef = 1, then SEARCH-LAYER on layer 0 with ef = max(ef_search, k);
    //   return the k closest, sorted ascending.
    throw std::logic_error("HnswIndex::search: not implemented");
}

}  // namespace hnsw
