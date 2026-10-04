#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

// On-disk format of a saved HnswIndex (version 1). Little-endian only.
//
//   offset 0        Header, 128 bytes
//   vectors_off     count * dim floats                    vector of node id at id * dim
//   levels_off      count bytes                           top layer of node id
//   level0_off      count * (1 + 2M) uint32               [neighbor count, ids...] per node
//   upper_offset_off count uint32                         slot offset into the arena, 0 = none
//   arena_off       arena_slots uint32                    layer 1+ blocks of (1 + M) slots
//
// Every section starts on a 128-byte boundary, which is a cache line on Apple Silicon and a
// multiple of the alignment any element type needs. The file is mapped and used in place, so
// its bytes must mean the same thing as the in-memory arrays they replace.
namespace hnsw::index_file {

constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'I', 'D', 'X', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304;  // reads back differently on a big-endian host
constexpr uint64_t kAlign = 128;
constexpr uint32_t kNoEntry = UINT32_MAX;

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t endian_marker;
    uint32_t header_size;
    uint32_t dim;
    uint32_t M;
    uint32_t use_heuristic;
    uint64_t ef_construction;
    uint64_t seed;
    uint64_t count;        // nodes stored
    uint64_t arena_slots;  // slots stored in the arena, including the unused slot 0
    uint32_t entry_id;     // kNoEntry when count == 0
    int32_t entry_level;   // -1 when count == 0
    uint64_t file_size;
    uint64_t vectors_off;
    uint64_t levels_off;
    uint64_t level0_off;
    uint64_t upper_offset_off;
    uint64_t arena_off;
    uint32_t header_crc;  // CRC-32 of every byte before this field
    uint32_t reserved;    // zero
};
static_assert(sizeof(Header) == 128, "Header has no padding and fills one alignment unit");
static_assert(std::is_trivially_copyable_v<Header>);

struct Layout {
    uint64_t vectors_off, levels_off, level0_off, upper_offset_off, arena_off, file_size;
};

inline uint64_t align_up(uint64_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }

// Where each section goes for the given sizes. Used by save to write and by load to check.
inline Layout compute_layout(uint64_t count, uint64_t dim, uint64_t M, uint64_t arena_slots) {
    Layout l{};
    l.vectors_off = align_up(sizeof(Header));
    l.levels_off = align_up(l.vectors_off + count * dim * sizeof(float));
    l.level0_off = align_up(l.levels_off + count);
    l.upper_offset_off = align_up(l.level0_off + count * (1 + 2 * M) * sizeof(uint32_t));
    l.arena_off = align_up(l.upper_offset_off + count * sizeof(uint32_t));
    l.file_size = align_up(l.arena_off + arena_slots * sizeof(uint32_t));
    return l;
}

inline uint32_t crc32(const void* data, size_t bytes) {
    uint32_t crc = 0xFFFFFFFFu;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

inline uint32_t header_crc(const Header& h) { return crc32(&h, offsetof(Header, header_crc)); }

// Cheap structural checks on a header against the real file size. Returns an empty string when
// the header is usable, otherwise the first problem found. It reads no node data, so it runs in
// microseconds. It cannot catch a damaged link, only HnswIndex::LoadOptions::verify does.
inline std::string validate(const Header& h, uint64_t actual_file_size) {
    if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0) return "not an HNSW index file (bad magic)";
    if (h.endian_marker != kEndianMarker) return "written on a machine with a different byte order";
    if (h.version != kVersion) {
        return "unsupported format version " + std::to_string(h.version) + " (this build reads " +
               std::to_string(kVersion) + ")";
    }
    if (h.header_size != sizeof(Header)) return "unexpected header size";
    if (h.header_crc != header_crc(h)) return "header checksum mismatch (file damaged)";
    if (h.reserved != 0) return "reserved header field is not zero";
    if (h.dim == 0 || h.dim > (1u << 24)) return "dimension out of range";
    if (h.M < 2 || h.M > (1u << 16)) return "M out of range";
    if (h.count >= UINT32_MAX) return "node count out of range";
    if (h.arena_slots < 1 || h.arena_slots >= UINT32_MAX) return "arena size out of range";
    if (h.count == 0) {
        if (h.entry_id != kNoEntry || h.entry_level != -1) return "empty index has an entry node";
    } else if (h.entry_id >= h.count || h.entry_level < 0 || h.entry_level > 255) {
        return "entry node out of range";
    }
    if (h.file_size != actual_file_size) {
        return "file size " + std::to_string(actual_file_size) + " does not match the header (" +
               std::to_string(h.file_size) + "), file is truncated or extended";
    }
    const Layout l = compute_layout(h.count, h.dim, h.M, h.arena_slots);
    if (l.file_size != h.file_size || l.vectors_off != h.vectors_off ||
        l.levels_off != h.levels_off || l.level0_off != h.level0_off ||
        l.upper_offset_off != h.upper_offset_off || l.arena_off != h.arena_off) {
        return "section offsets do not match the stored sizes";
    }
    return {};
}

}  // namespace hnsw::index_file
