#pragma once

#include <cstdint>
#include <vector>

#include "hnsw/dataset.hpp"

namespace hnsw {

// Mean over queries of |returned[:k] ∩ groundtruth[:k]| / k.
// results[q] holds the ids returned for query q. Throws std::invalid_argument if
// results.size() != gt.count or k > gt.dim.
double recall_at_k(const std::vector<std::vector<uint32_t>>& results, const IvecsData& gt,
                   size_t k);

struct LatencyStats {
    double mean_us = 0;
    double p50_us = 0;
    double p99_us = 0;
};

// Nearest-rank percentiles over per-query latencies (microseconds). Input is copied.
LatencyStats summarize_latencies(std::vector<double> latencies_us);

}  // namespace hnsw
