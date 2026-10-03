#include "hnsw/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hnsw {

double recall_at_k(const std::vector<std::vector<uint32_t>>& results, const IvecsData& gt,
                   size_t k) {
    if (results.size() != gt.count) {
        throw std::invalid_argument("recall_at_k: results has " + std::to_string(results.size()) +
                                    " queries but ground truth has " + std::to_string(gt.count));
    }
    if (k == 0 || k > gt.dim) {
        throw std::invalid_argument("recall_at_k: k=" + std::to_string(k) +
                                    " must be in [1, ground-truth width " + std::to_string(gt.dim) +
                                    "]");
    }
    if (gt.count == 0) return 0.0;

    size_t hits = 0;
    for (size_t q = 0; q < gt.count; ++q) {
        const int32_t* truth = gt.row(q);
        const size_t n = std::min(k, results[q].size());
        for (size_t i = 0; i < n; ++i) {
            const auto id = static_cast<int32_t>(results[q][i]);
            if (std::find(truth, truth + k, id) != truth + k) ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(gt.count * k);
}

LatencyStats summarize_latencies(std::vector<double> v) {
    LatencyStats s;
    if (v.empty()) return s;
    s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {  // nearest-rank
        const auto rank = static_cast<size_t>(std::ceil(p * static_cast<double>(v.size())));
        return v[std::clamp<size_t>(rank, 1, v.size()) - 1];
    };
    s.p50_us = pct(0.50);
    s.p99_us = pct(0.99);
    return s;
}

}  // namespace hnsw
