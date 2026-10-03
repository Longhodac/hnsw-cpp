// Eval CLI: load a TEXMEX dataset, build an index, run every query, report metrics,
// and append one CSV row per ef_search value.
//
//   eval --dataset data/siftsmall --index bruteforce
//   eval --dataset data/sift --index hnsw --M 16 --ef-construction 200 --ef-search 10,20,50,100

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hnsw/brute_force.hpp"
#include "hnsw/dataset.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/metrics.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

void usage() {
    std::fputs(
        "usage: eval --dataset <dir> [options]\n"
        "  <dir> holds <name>_base.fvecs, <name>_query.fvecs, <name>_groundtruth.ivecs,\n"
        "  where <name> is the directory's basename (e.g. data/siftsmall).\n"
        "options:\n"
        "  --index bruteforce|hnsw   index to run (default bruteforce)\n"
        "  --k N                     neighbors per query (default 10)\n"
        "  --M N                     HNSW max connections (default 16)\n"
        "  --ef-construction N       HNSW build beam width (default 200)\n"
        "  --ef-search A[,B,...]     HNSW search beam width(s); one run + CSV row each (default 50)\n"
        "  --seed N                  RNG seed (default 42)\n"
        "  --max-queries N           use only the first N queries (default all)\n"
        "  --csv PATH                append results here (default results/results.csv; 'none' disables)\n",
        stderr);
}

std::vector<size_t> parse_list(const std::string& s) {
    std::vector<size_t> out;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');) out.push_back(std::stoull(tok));
    return out;
}

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> opt = {{"index", "bruteforce"}, {"k", "10"},
                                              {"M", "16"},            {"ef-construction", "200"},
                                              {"ef-search", "50"},    {"seed", "42"},
                                              {"max-queries", "0"},   {"csv", "results/results.csv"}};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        }
        const bool known = a.rfind("--", 0) == 0 && (a == "--dataset" || opt.contains(a.substr(2)));
        if (!known || i + 1 >= argc) {
            std::fprintf(stderr, "eval: bad argument '%s'\n", a.c_str());
            usage();
            return 2;
        }
        opt[a.substr(2)] = argv[++i];
    }
    if (!opt.contains("dataset")) {
        usage();
        return 2;
    }

    try {
        const fs::path dir = fs::path(opt["dataset"]);
        std::string name = dir.filename().string();
        if (name.empty()) name = dir.parent_path().filename().string();
        const auto base = hnsw::load_fvecs((dir / (name + "_base.fvecs")).string());
        const auto queries = hnsw::load_fvecs((dir / (name + "_query.fvecs")).string());
        const auto gt = hnsw::load_ivecs((dir / (name + "_groundtruth.ivecs")).string());

        if (queries.dim != base.dim) {
            throw std::runtime_error("query dim " + std::to_string(queries.dim) +
                                     " != base dim " + std::to_string(base.dim));
        }
        if (gt.count != queries.count) {
            throw std::runtime_error("ground truth has " + std::to_string(gt.count) +
                                     " rows but there are " + std::to_string(queries.count) +
                                     " queries");
        }
        const size_t k = std::stoull(opt["k"]);
        size_t nq = queries.count;
        if (const size_t cap = std::stoull(opt["max-queries"]); cap > 0 && cap < nq) nq = cap;

        // Restrict ground truth to the nq queries we run so recall matches.
        const hnsw::IvecsData gt_used{
            gt.dim, nq, {gt.data.begin(), gt.data.begin() + static_cast<std::ptrdiff_t>(nq * gt.dim)}};

        const size_t M = std::stoull(opt["M"]);
        const size_t efc = std::stoull(opt["ef-construction"]);
        const uint64_t seed = std::stoull(opt["seed"]);
        const bool is_hnsw = opt["index"] == "hnsw";

        std::unique_ptr<hnsw::Index> index;
        if (is_hnsw) {
            index = std::make_unique<hnsw::HnswIndex>(base.dim, M, efc, base.count, seed);
        } else if (opt["index"] == "bruteforce") {
            index = std::make_unique<hnsw::BruteForceIndex>(base.dim);
        } else {
            throw std::runtime_error("unknown --index '" + opt["index"] + "'");
        }
        const std::string index_name(index->name());

        std::printf("dataset=%s n=%zu nq=%zu dim=%zu index=%s k=%zu\n", name.c_str(), base.count,
                    nq, base.dim, index_name.c_str(), k);

        const auto t0 = Clock::now();
        for (size_t i = 0; i < base.count; ++i) index->add(base.row(i));
        const double build_s = seconds(t0, Clock::now());
        std::printf("build_s=%.3f\n", build_s);

        // Exact indexes ignore ef_search, so run them once.
        const std::vector<size_t> efs =
            is_hnsw ? parse_list(opt["ef-search"]) : std::vector<size_t>{0};

        std::ofstream csv;
        if (opt["csv"] != "none") {
            const fs::path p(opt["csv"]);
            if (p.has_parent_path()) fs::create_directories(p.parent_path());
            const bool fresh = !fs::exists(p) || fs::file_size(p) == 0;
            csv.open(p, std::ios::app);
            if (!csv) throw std::runtime_error("cannot open CSV " + p.string());
            if (fresh) {
                csv << "dataset,index,n,nq,dim,k,M,ef_construction,ef_search,seed,build_s,"
                       "recall,qps,mean_us,p50_us,p99_us\n";
            }
        }

        for (const size_t ef : efs) {
            std::vector<std::vector<uint32_t>> results(nq);
            std::vector<double> lat_us(nq);
            const auto q0 = Clock::now();
            for (size_t q = 0; q < nq; ++q) {
                const auto s = Clock::now();
                auto nn = index->search(queries.row(q), k, ef);
                const auto e = Clock::now();
                lat_us[q] = std::chrono::duration<double, std::micro>(e - s).count();
                results[q].reserve(nn.size());
                for (const auto& [dist, id] : nn) results[q].push_back(id);
            }
            const double total_s = seconds(q0, Clock::now());

            const double recall = hnsw::recall_at_k(results, gt_used, k);
            const double qps = static_cast<double>(nq) / total_s;
            const auto lat = hnsw::summarize_latencies(std::move(lat_us));

            std::printf("ef_search=%zu recall@%zu=%.4f qps=%.1f mean_us=%.2f p50_us=%.2f p99_us=%.2f\n",
                        ef, k, recall, qps, lat.mean_us, lat.p50_us, lat.p99_us);
            if (csv.is_open()) {
                char row[512];
                std::snprintf(row, sizeof row,
                              "%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%llu,%.4f,%.6f,%.2f,%.3f,%.3f,%.3f\n",
                              name.c_str(), index_name.c_str(), base.count, nq, base.dim, k,
                              is_hnsw ? M : 0, is_hnsw ? efc : 0, ef,
                              static_cast<unsigned long long>(seed), build_s, recall, qps,
                              lat.mean_us, lat.p50_us, lat.p99_us);
                csv << row;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "eval: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
