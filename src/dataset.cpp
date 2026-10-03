#include "hnsw/dataset.hpp"

#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace hnsw {
namespace {

static_assert(std::endian::native == std::endian::little,
              "fvecs/ivecs are little-endian; add byte swapping for big-endian hosts");

constexpr int32_t kMaxDim = 1 << 20;  // sanity bound; catches non-vecs files early

struct FileCloser {
    void operator()(std::FILE* f) const { std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

[[noreturn]] void fail(const std::string& path, const std::string& msg) {
    throw std::runtime_error(path + ": " + msg);
}

template <typename T>
VecSet<T> load_vecs(const std::string& path) {
    static_assert(sizeof(T) == sizeof(int32_t));

    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) fail(path, std::string("cannot open file: ") + std::strerror(errno));

    if (std::fseek(f.get(), 0, SEEK_END) != 0) fail(path, "cannot determine file size");
    const long end = std::ftell(f.get());
    if (end < 0) fail(path, "cannot determine file size");
    const auto file_size = static_cast<uint64_t>(end);
    std::rewind(f.get());

    if (file_size == 0) fail(path, "file is empty");
    if (file_size < sizeof(int32_t)) fail(path, "file too short to hold a dimension header");

    int32_t dim = 0;
    if (std::fread(&dim, sizeof(dim), 1, f.get()) != 1) fail(path, "failed to read header");
    if (dim <= 0 || dim > kMaxDim) {
        fail(path, "invalid dimension " + std::to_string(dim) + " in record 0 (not a .vecs file?)");
    }

    const uint64_t record_bytes = sizeof(int32_t) + static_cast<uint64_t>(dim) * sizeof(T);
    if (file_size % record_bytes != 0) {
        fail(path, "file size " + std::to_string(file_size) + " is not a multiple of the record size " +
                       std::to_string(record_bytes) + " (dim=" + std::to_string(dim) +
                       "); file is truncated or corrupt");
    }

    VecSet<T> out;
    out.dim = static_cast<size_t>(dim);
    out.count = static_cast<size_t>(file_size / record_bytes);
    out.data.resize(out.count * out.dim);

    std::rewind(f.get());
    for (size_t i = 0; i < out.count; ++i) {
        int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, f.get()) != 1) {
            fail(path, "unexpected EOF reading record " + std::to_string(i));
        }
        if (d != dim) {
            fail(path, "record " + std::to_string(i) + " has dimension " + std::to_string(d) +
                           ", expected " + std::to_string(dim));
        }
        if (std::fread(out.data.data() + i * out.dim, sizeof(T), out.dim, f.get()) != out.dim) {
            fail(path, "unexpected EOF reading record " + std::to_string(i));
        }
    }
    return out;
}

}  // namespace

FvecsData load_fvecs(const std::string& path) { return load_vecs<float>(path); }
IvecsData load_ivecs(const std::string& path) { return load_vecs<int32_t>(path); }

}  // namespace hnsw
