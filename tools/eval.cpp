// Eval CLI: load a TEXMEX dataset, build an index, run every query, report metrics,
// and append one CSV row per ef_search value.
//
//   eval --dataset data/siftsmall --index bruteforce
//   eval --dataset data/sift --index hnsw --M 16 --ef-construction 200 --ef-search 10,20,50,100

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
        "  --heuristic 0|1           HNSW neighbor selection: 1 = Algorithm 4 (default), 0 = M closest\n"
        "  --seed N                  RNG seed (default 42)\n"
        "  --query-repeat R          run the query set R times for timing; recall and latency\n"
        "                            use the first pass only (default 1)\n"
        "  --threads N               build with N threads (hnsw only; default 1)\n"
        "  --search-threads A[,B,...] query threads; one run + CSV row per (ef, threads) (default 1)\n"
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
                                              {"ef-search", "50"},    {"seed", "42"},       {"heuristic", "1"},
                                              {"threads", "1"},       {"search-threads", "1"}, {"query-repeat", "1"},
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
        const bool is_hnsw = opt["index"] == "hnsw";  // "hnsw-simple" in output = heuristic off

        std::unique_ptr<hnsw::Index> index;
        if (is_hnsw) {
            auto h = std::make_unique<hnsw::HnswIndex>(base.dim, M, efc, base.count, seed);
            h->set_use_heuristic(opt["heuristic"] != "0");
            index = std::move(h);
        } else if (opt["index"] == "bruteforce") {
            index = std::make_unique<hnsw::BruteForceIndex>(base.dim);
        } else {
            throw std::runtime_error("unknown --index '" + opt["index"] + "'");
        }
        const std::string index_name(index->name());

        std::printf("dataset=%s n=%zu nq=%zu dim=%zu index=%s k=%zu\n", name.c_str(), base.count,
                    nq, base.dim, index_name.c_str(), k);

        const size_t threads = std::stoull(opt["threads"]);
        if (threads == 0) throw std::runtime_error("--threads must be >= 1");
        if (threads > 1 && !is_hnsw) throw std::runtime_error("--threads > 1 needs --index hnsw");

        // A parallel build hands out ids in a thread-dependent order, but the ground truth
        // names rows of the base file. Remember each id's row and translate results back.
        std::vector<uint32_t> row_of_id(base.count);
        const auto t0 = Clock::now();
        if (threads == 1) {
            for (size_t i = 0; i < base.count; ++i) {
                row_of_id[index->add(base.row(i))] = static_cast<uint32_t>(i);
            }
        } else {
            std::atomic<size_t> next_row{0};
            std::exception_ptr failure;
            std::mutex failure_mutex;
            std::vector<std::thread> pool;
            for (size_t t = 0; t < threads; ++t) {
                pool.emplace_back([&] {
                    try {
                        for (size_t i; (i = next_row.fetch_add(1)) < base.count;) {
                            row_of_id[index->add(base.row(i))] = static_cast<uint32_t>(i);
                        }
                    } catch (...) {
                        const std::lock_guard<std::mutex> lock(failure_mutex);
                        if (!failure) failure = std::current_exception();
                        next_row = base.count;  // stop the other workers
                    }
                });
            }
            for (auto& th : pool) th.join();
            if (failure) std::rethrow_exception(failure);
        }
        const double build_s = seconds(t0, Clock::now());
        std::printf("build_s=%.3f build_threads=%zu\n", build_s, threads);

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
                       "recall,qps,mean_us,p50_us,p99_us,build_threads,search_threads\n";
            }
        }

        const std::vector<size_t> search_threads = parse_list(opt["search-threads"]);
        const size_t repeat = std::stoull(opt["query-repeat"]);
        if (repeat == 0) throw std::runtime_error("--query-repeat must be >= 1");
        for (const size_t st : search_threads) {
            if (st == 0) throw std::runtime_error("--search-threads values must be >= 1");
            if (st > 1 && !is_hnsw) throw std::runtime_error("--search-threads > 1 needs --index hnsw");
        }

        for (const size_t ef : efs) {
            for (const size_t st : search_threads) {
                std::vector<std::vector<uint32_t>> results(nq);
                std::vector<double> lat_us(nq);
                std::atomic<size_t> next_query{0};
                auto work = [&] {
                    for (size_t j; (j = next_query.fetch_add(1)) < nq * repeat;) {
                        const size_t q = j % nq;
                        const auto s = Clock::now();
                        auto nn = index->search(queries.row(q), k, ef);
                        const auto e = Clock::now();
                        if (j >= nq) continue;  // later passes only add to the timed work
                        lat_us[q] = std::chrono::duration<double, std::micro>(e - s).count();
                        results[q].reserve(nn.size());
                        for (const auto& [dist, id] : nn) results[q].push_back(row_of_id[id]);
                    }
                };
                const auto q0 = Clock::now();
                if (st == 1) {
                    work();
                } else {
                    std::vector<std::thread> pool;
                    for (size_t t = 0; t < st; ++t) pool.emplace_back(work);
                    for (auto& th : pool) th.join();
                }
                const double total_s = seconds(q0, Clock::now());

                const double recall = hnsw::recall_at_k(results, gt_used, k);
                const double qps = static_cast<double>(nq * repeat) / total_s;
                const auto lat = hnsw::summarize_latencies(std::move(lat_us));

                std::printf("ef_search=%zu search_threads=%zu recall@%zu=%.4f qps=%.1f mean_us=%.2f p50_us=%.2f p99_us=%.2f\n",
                            ef, st, k, recall, qps, lat.mean_us, lat.p50_us, lat.p99_us);
                if (csv.is_open()) {
                    char row[512];
                    std::snprintf(row, sizeof row,
                                  "%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%llu,%.4f,%.6f,%.2f,%.3f,%.3f,%.3f,%zu,%zu\n",
                                  name.c_str(), index_name.c_str(), base.count, nq, base.dim, k,
                                  is_hnsw ? M : 0, is_hnsw ? efc : 0, ef,
                                  static_cast<unsigned long long>(seed), build_s, recall, qps,
                                  lat.mean_us, lat.p50_us, lat.p99_us, threads, st);
                    csv << row;
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "eval: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
