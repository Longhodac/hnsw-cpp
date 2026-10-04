#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

#include "hnsw/memory_region.hpp"

using hnsw::MemoryRegion;

TEST(MemoryRegion, AnonymousIsZeroedAndWritable) {
    const size_t bytes = size_t{1} << 20;
    MemoryRegion r = MemoryRegion::anonymous(bytes);
    ASSERT_NE(r.data(), nullptr);
    EXPECT_EQ(r.size(), bytes);
    auto* p = static_cast<uint8_t*>(r.data());
    for (size_t i = 0; i < bytes; i += 4099) EXPECT_EQ(p[i], 0) << "offset " << i;
    p[0] = 7;
    p[bytes - 1] = 9;
    EXPECT_EQ(p[0], 7);
    EXPECT_EQ(p[bytes - 1], 9);
}

TEST(MemoryRegion, DataIsPageAligned) {
    MemoryRegion r = MemoryRegion::anonymous(100);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(r.data()) % 4096, 0u);
}

TEST(MemoryRegion, ZeroBytesGivesEmptyRegion) {
    const MemoryRegion r = MemoryRegion::anonymous(0);
    EXPECT_EQ(r.data(), nullptr);
    EXPECT_EQ(r.size(), 0u);
}

TEST(MemoryRegion, MoveTransfersOwnership) {
    MemoryRegion a = MemoryRegion::anonymous(4096);
    void* where = a.data();
    MemoryRegion b = std::move(a);
    EXPECT_EQ(b.data(), where);
    EXPECT_EQ(b.size(), 4096u);
    EXPECT_EQ(a.data(), nullptr);  // NOLINT(bugprone-use-after-move): moved-from is empty by contract
    EXPECT_EQ(a.size(), 0u);

    MemoryRegion c = MemoryRegion::anonymous(8192);
    c = std::move(b);  // releases c's old mapping, takes b's
    EXPECT_EQ(c.data(), where);
    EXPECT_EQ(c.size(), 4096u);
}
