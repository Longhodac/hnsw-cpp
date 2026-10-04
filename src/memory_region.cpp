#include "hnsw/memory_region.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace hnsw {

MemoryRegion::~MemoryRegion() { reset(); }

MemoryRegion::MemoryRegion(MemoryRegion&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MemoryRegion& MemoryRegion::operator=(MemoryRegion&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

void MemoryRegion::reset() noexcept {
    if (data_ != nullptr) munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
}

MemoryRegion MemoryRegion::anonymous(size_t bytes) {
    MemoryRegion region;
    if (bytes == 0) return region;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    region.data_ = p;
    region.size_ = bytes;
    return region;
}

MemoryRegion MemoryRegion::map_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("cannot stat " + path + ": " + std::strerror(err));
    }
    MemoryRegion region;
    if (st.st_size > 0) {
        void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        const int err = errno;
        ::close(fd);
        if (p == MAP_FAILED) {
            throw std::runtime_error("cannot map " + path + ": " + std::strerror(err));
        }
        region.data_ = p;
        region.size_ = static_cast<size_t>(st.st_size);
    } else {
        ::close(fd);
    }
    return region;
}

}  // namespace hnsw
