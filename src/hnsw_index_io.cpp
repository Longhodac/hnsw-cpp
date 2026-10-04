// Saving and loading HnswIndex. The file format is documented in hnsw/index_file.hpp.

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "hnsw/hnsw_index.hpp"
#include "hnsw/index_file.hpp"

namespace hnsw {

namespace {

namespace file = index_file;

std::string errno_text() { return std::strerror(errno); }

// Closes the descriptor on every path out of save().
struct FdGuard {
    int fd;
    ~FdGuard() {
        if (fd >= 0) ::close(fd);
    }
};

void write_at(int fd, const void* data, size_t bytes, uint64_t offset, const char* what) {
    constexpr size_t kMaxChunk = size_t{1} << 30;  // macOS rejects writes above 2 GiB
    const char* p = static_cast<const char*>(data);
    while (bytes > 0) {
        const ssize_t n = ::pwrite(fd, p, std::min(bytes, kMaxChunk), static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("save: writing ") + what + " failed: " + errno_text());
        }
        p += n;
        offset += static_cast<uint64_t>(n);
        bytes -= static_cast<size_t>(n);
    }
}

// Flushes file contents to the device. fsync alone does not reach the platters on macOS.
void sync_fd(int fd) {
#ifdef F_FULLFSYNC
    if (::fcntl(fd, F_FULLFSYNC) == 0) return;
#endif
    if (::fsync(fd) != 0) throw std::runtime_error("save: fsync failed: " + errno_text());
}

}  // namespace

void HnswIndex::save(const std::string& path) const {
    const size_t n = count_.load();
    const Entry entry = load_entry();
    const size_t arena_slots = std::min(upper_arena_used_.load(), upper_arena_capacity_);
    const file::Layout layout = file::compute_layout(n, dim_, M_, arena_slots);

    file::Header h;
    std::memset(&h, 0, sizeof h);  // zero the padding-free struct so the checksum is stable
    std::memcpy(h.magic, file::kMagic, sizeof h.magic);
    h.version = file::kVersion;
    h.endian_marker = file::kEndianMarker;
    h.header_size = sizeof(file::Header);
    h.dim = static_cast<uint32_t>(dim_);
    h.M = static_cast<uint32_t>(M_);
    h.use_heuristic = use_heuristic_ ? 1u : 0u;
    h.ef_construction = ef_construction_;
    h.seed = seed_;
    h.count = n;
    h.arena_slots = arena_slots;
    h.entry_id = entry.id;
    h.entry_level = entry.level;
    h.file_size = layout.file_size;
    h.vectors_off = layout.vectors_off;
    h.levels_off = layout.levels_off;
    h.level0_off = layout.level0_off;
    h.upper_offset_off = layout.upper_offset_off;
    h.arena_off = layout.arena_off;
    h.header_crc = file::header_crc(h);

    const std::string tmp = path + ".tmp";
    FdGuard guard{::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644)};
    if (guard.fd < 0) throw std::runtime_error("save: cannot create " + tmp + ": " + errno_text());
    try {
        // Sizing the file first leaves the padding between sections as zeros.
        if (::ftruncate(guard.fd, static_cast<off_t>(layout.file_size)) != 0) {
            throw std::runtime_error("save: cannot size " + tmp + ": " + errno_text());
        }
        write_at(guard.fd, &h, sizeof h, 0, "header");
        write_at(guard.fd, vectors_, n * dim_ * sizeof(float), layout.vectors_off, "vectors");
        write_at(guard.fd, levels_, n, layout.levels_off, "levels");
        write_at(guard.fd, level0_, n * (1 + 2 * M_) * sizeof(uint32_t), layout.level0_off,
                 "layer 0 links");
        write_at(guard.fd, upper_offset_, n * sizeof(uint32_t), layout.upper_offset_off,
                 "upper offsets");
        write_at(guard.fd, upper_arena_, arena_slots * sizeof(uint32_t), layout.arena_off,
                 "upper link arena");
        sync_fd(guard.fd);
        if (::close(std::exchange(guard.fd, -1)) != 0) {
            throw std::runtime_error("save: close failed: " + errno_text());
        }
        if (::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("save: cannot rename " + tmp + " to " + path + ": " +
                                     errno_text());
        }
    } catch (...) {
        ::unlink(tmp.c_str());
        throw;
    }

    // Make the rename itself durable. A failure here leaves a complete file, so ignore it.
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (dir.empty()) dir = ".";
    const int dir_fd = ::open(dir.c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }
}

HnswIndex::HnswIndex(LoadTag, MemoryRegion mapped, const file::Header& h)
    : dim_(h.dim),
      M_(h.M),
      max_elements_(h.count),
      seed_(h.seed),
      ef_construction_(h.ef_construction),
      mL_(1.0 / std::log(static_cast<double>(h.M))),
      file_region_(std::move(mapped)),
      read_only_(true) {
    auto* base = static_cast<char*>(file_region_.data());
    vectors_ = reinterpret_cast<float*>(base + h.vectors_off);
    levels_ = reinterpret_cast<uint8_t*>(base + h.levels_off);
    level0_ = reinterpret_cast<uint32_t*>(base + h.level0_off);
    upper_offset_ = reinterpret_cast<uint32_t*>(base + h.upper_offset_off);
    upper_arena_ = reinterpret_cast<uint32_t*>(base + h.arena_off);
    upper_arena_capacity_ = h.arena_slots;
    upper_arena_used_ = h.arena_slots;
    count_ = h.count;
    use_heuristic_ = h.use_heuristic != 0;
    entry_.store(pack({h.entry_id, h.entry_level}), std::memory_order_release);
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path) {
    return load(path, LoadOptions{});
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path, LoadOptions options) {
    MemoryRegion mapped = MemoryRegion::map_file(path);
    if (mapped.size() < sizeof(file::Header)) {
        throw std::runtime_error("load: " + path + ": file is too small to be an HNSW index");
    }
    file::Header h;
    std::memcpy(&h, mapped.data(), sizeof h);
    if (const std::string problem = file::validate(h, mapped.size()); !problem.empty()) {
        throw std::runtime_error("load: " + path + ": " + problem);
    }

    std::unique_ptr<HnswIndex> index(new HnswIndex(LoadTag{}, std::move(mapped), h));
    if (h.count > 0 && index->levels_[h.entry_id] != h.entry_level) {
        throw std::runtime_error("load: " + path + ": entry node does not match its stored level");
    }
    if (options.verify) index->verify_structure();
    return index;
}

void HnswIndex::verify_structure() const {
    const size_t n = count_.load();
    const Entry entry = load_entry();
    const size_t arena_slots = std::min(upper_arena_used_.load(), upper_arena_capacity_);
    auto fail = [](uint32_t id, int layer, const std::string& what) {
        throw std::runtime_error("verify: node " + std::to_string(id) + " layer " +
                                 std::to_string(layer) + ": " + what);
    };

    if (n == 0) {
        if (entry.id != kInvalidId) fail(entry.id, 0, "empty index has an entry node");
        return;
    }
    if (entry.id >= n) fail(entry.id, 0, "entry node is out of range");
    if (levels_[entry.id] != entry.level) fail(entry.id, 0, "entry level disagrees with levels");

    for (uint32_t id = 0; id < n; ++id) {
        const int level = levels_[id];
        if (level > entry.level) fail(id, level, "level above the entry node's level");
        if (level > 0) {
            const size_t offset = upper_offset_[id];
            if (offset < 1 || offset + static_cast<size_t>(level) * (1 + M_) > arena_slots) {
                fail(id, level, "upper offset is outside the arena");
            }
        } else if (upper_offset_[id] != 0) {
            fail(id, 0, "level-0 node has an upper offset");
        }
        for (int layer = 0; layer <= level; ++layer) {
            const uint32_t* block = link_block(id, layer);
            const size_t count = block[0];
            if (count > max_degree(layer)) fail(id, layer, "neighbor count above the limit");
            for (size_t i = 0; i < count; ++i) {
                const uint32_t nbr = block[1 + i];
                if (nbr >= n) fail(id, layer, "link to a node that does not exist");
                if (nbr == id) fail(id, layer, "link to itself");
                if (levels_[nbr] < layer) fail(id, layer, "link to a node not on this layer");
            }
        }
    }
}

}  // namespace hnsw
