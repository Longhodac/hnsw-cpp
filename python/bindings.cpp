// Python module `hnsw_cpp`: a thin batch interface over hnsw::HnswIndex.
//
//   import hnsw_cpp
//   index = hnsw_cpp.HnswIndex(dim=128, max_elements=1_000_000, M=16, ef_construction=100)
//   ids = index.add_batch(base, threads=10)                  # base: float32 (n, dim), C-contiguous
//   labels, dists = index.search_batch(queries, k=10, ef=60, threads=10)
//
// Arrays must already be float32 and C-contiguous. The module never converts or copies an input
// for you, because a silent float64 to float32 copy would distort timings and hide a bug.
// Search returns the index's own ids. add_batch returns the id each row got, so a caller maps
// ids back to rows with one numpy index (see scripts/bench_common.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hnsw/hnsw_index.hpp"

namespace py = pybind11;
using hnsw::HnswIndex;

namespace {

// Checks that `a` is a float32, C-contiguous matrix with `dim` columns. Returns its row count.
size_t check_matrix(const py::array& a, const char* name, size_t dim) {
    if (!py::array_t<float>::check_(a)) {
        throw py::value_error(std::string(name) + " must be a float32 numpy array, got dtype " +
                              py::str(a.dtype()).cast<std::string>() +
                              " (convert with .astype(np.float32); nothing is converted for you)");
    }
    if (a.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be 2-dimensional (rows x dim), got " +
                              std::to_string(a.ndim()) + " dimension(s)");
    }
    if (!(a.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) +
                              " must be C-contiguous (use np.ascontiguousarray)");
    }
    if (static_cast<size_t>(a.shape(1)) != dim) {
        throw py::value_error(std::string(name) + " has " + std::to_string(a.shape(1)) +
                              " columns but the index has dimension " + std::to_string(dim));
    }
    return static_cast<size_t>(a.shape(0));
}

void check_threads(size_t threads) {
    if (threads == 0) throw py::value_error("threads must be at least 1");
}

// Runs fn(0) ... fn(items - 1) on up to `threads` threads, the calling thread included. Items
// are handed out one at a time, so uneven work balances itself. fn must not touch Python
// objects, since the GIL is released. The first exception any worker throws is rethrown after
// all workers have stopped.
template <typename Fn>
void run_parallel(size_t items, size_t threads, Fn fn) {
    threads = std::min(threads, items);
    if (threads <= 1) {
        for (size_t i = 0; i < items; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto worker = [&] {
        try {
            for (size_t i; (i = next.fetch_add(1)) < items;) fn(i);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            next = items;  // tell the other workers to stop
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    if (failure) std::rethrow_exception(failure);
}

py::array_t<uint32_t> add_batch(HnswIndex& index, const py::array& vectors, size_t threads) {
    check_threads(threads);
    const size_t n = check_matrix(vectors, "vectors", index.dim());
    if (index.read_only()) throw std::logic_error("this index was loaded from a file and is read-only");
    const size_t room = index.capacity() - std::min(index.size(), index.capacity());
    if (n > room) {
        throw py::value_error("adding " + std::to_string(n) + " vectors would exceed the capacity of " +
                              std::to_string(index.capacity()) + " (size " +
                              std::to_string(index.size()) + "); nothing was added");
    }

    py::array_t<uint32_t> ids(static_cast<py::ssize_t>(n));
    uint32_t* const out = ids.mutable_data();
    const float* const data = static_cast<const float*>(vectors.data());
    const size_t dim = index.dim();
    {
        py::gil_scoped_release release;
        run_parallel(n, threads, [&](size_t i) { out[i] = index.add(data + i * dim); });
    }
    return ids;
}

py::tuple search_batch(const HnswIndex& index, const py::array& queries, size_t k, size_t ef,
                       size_t threads) {
    check_threads(threads);
    if (k == 0) throw py::value_error("k must be at least 1");
    if (ef == 0) throw py::value_error("ef must be at least 1");
    const size_t m = check_matrix(queries, "queries", index.dim());

    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(k)};
    py::array_t<int64_t> ids(shape);
    py::array_t<float> dists(shape);
    int64_t* const id_out = ids.mutable_data();
    float* const dist_out = dists.mutable_data();
    const float* const data = static_cast<const float*>(queries.data());
    const size_t dim = index.dim();
    {
        py::gil_scoped_release release;
        run_parallel(m, threads, [&](size_t q) {
            const auto found = index.search(data + q * dim, k, ef);
            for (size_t j = 0; j < k; ++j) {
                const bool have = j < found.size();
                id_out[q * k + j] = have ? static_cast<int64_t>(found[j].second) : -1;
                dist_out[q * k + j] = have ? found[j].first : std::numeric_limits<float>::infinity();
            }
        });
    }
    return py::make_tuple(ids, dists);
}

}  // namespace

PYBIND11_MODULE(hnsw_cpp, m) {
    m.doc() = "Batch Python interface to the from-scratch C++ HNSW index.";

    py::class_<HnswIndex>(m, "HnswIndex",
                          "An HNSW index. add_batch and search_batch can run on many threads; a "
                          "loaded index is read-only.")
        .def(py::init([](size_t dim, size_t max_elements, size_t M, size_t ef_construction,
                         uint64_t seed, bool use_heuristic) {
                 auto index = std::make_unique<HnswIndex>(dim, M, ef_construction, max_elements, seed);
                 index->set_use_heuristic(use_heuristic);
                 return index;
             }),
             py::arg("dim"), py::arg("max_elements"), py::arg("M") = 16,
             py::arg("ef_construction") = 100, py::arg("seed") = 42, py::arg("use_heuristic") = true)
        .def("add_batch", &add_batch, py::arg("vectors"), py::arg("threads") = 1,
             "Adds the rows of a float32 (n, dim) C-contiguous array and returns the uint32 id "
             "each row received. With threads=1 the ids are 0, 1, 2, ... in row order. With more "
             "threads the ids are a permutation of that range.")
        .def("search_batch", &search_batch, py::arg("queries"), py::arg("k"), py::arg("ef"),
             py::arg("threads") = 1,
             "Returns (ids, squared distances) for each query row: int64 and float32 arrays of "
             "shape (m, k), nearest first. Missing results (an index smaller than k) are -1 and "
             "infinity.")
        .def("save", &HnswIndex::save, py::arg("path"), py::call_guard<py::gil_scoped_release>(),
             "Writes the index to a file. Nothing may add to the index meanwhile.")
        .def_static(
            "load",
            [](const std::string& path, bool verify) {
                py::gil_scoped_release release;
                return HnswIndex::load(path, HnswIndex::LoadOptions{.verify = verify});
            },
            py::arg("path"), py::arg("verify") = false,
            "Maps a saved index in place (read-only). verify=True also checks every link.")
        .def_property_readonly("size", &HnswIndex::size)
        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("capacity", &HnswIndex::capacity)
        .def_property_readonly("read_only", &HnswIndex::read_only)
        .def("__len__", &HnswIndex::size)
        .def("__repr__", [](const HnswIndex& index) {
            return "<hnsw_cpp.HnswIndex dim=" + std::to_string(index.dim()) +
                   " size=" + std::to_string(index.size()) +
                   " capacity=" + std::to_string(index.capacity()) +
                   (index.read_only() ? " read_only" : "") + ">";
        });
}
