#pragma once

#include <cstddef>
#include <string>

namespace hnsw {

// A range of memory owned through mmap. A region comes either from anonymous memory or from a
// mapped file. An anonymous region is zero-filled and only uses
// physical memory for the pages that get touched, so a large one costs almost nothing until
// it is written. Pages are 16 KiB on Apple Silicon, so data() is well aligned for any
// element type. Move-only; unmaps on destruction.
class MemoryRegion {
public:
    MemoryRegion() = default;
    ~MemoryRegion();
    MemoryRegion(MemoryRegion&& other) noexcept;
    MemoryRegion& operator=(MemoryRegion&& other) noexcept;
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion& operator=(const MemoryRegion&) = delete;

    // Zero-filled and writable. Throws std::bad_alloc if the mapping fails. bytes == 0 gives
    // an empty region with a null data().
    static MemoryRegion anonymous(size_t bytes);

    // Maps a whole file read-only. Pages load from disk on first touch, so this returns in
    // microseconds whatever the file size. Writing through data() faults. Throws
    // std::runtime_error if the file cannot be opened or mapped. An empty file gives an empty
    // region.
    static MemoryRegion map_file(const std::string& path);

    void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void reset() noexcept;

    void* data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace hnsw
