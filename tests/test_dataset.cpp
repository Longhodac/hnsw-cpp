#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "hnsw/dataset.hpp"

namespace fs = std::filesystem;

namespace {

fs::path tmp_path(const char* name) {
    return fs::temp_directory_path() / (std::string("hnsw_test_") + name);
}

template <typename T>
void write_vecs(const fs::path& p, const std::vector<std::vector<T>>& rows) {
    std::ofstream f(p, std::ios::binary);
    for (const auto& r : rows) {
        const auto d = static_cast<int32_t>(r.size());
        f.write(reinterpret_cast<const char*>(&d), sizeof d);
        f.write(reinterpret_cast<const char*>(r.data()),
                static_cast<std::streamsize>(r.size() * sizeof(T)));
    }
}

void patch_int32(const fs::path& p, std::streamoff offset, int32_t value) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(offset);
    f.write(reinterpret_cast<const char*>(&value), sizeof value);
}

std::string error_of(const fs::path& p) {
    try {
        hnsw::load_fvecs(p.string());
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST(Dataset, LoadsFvecs) {
    const auto p = tmp_path("ok.fvecs");
    write_vecs<float>(p, {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    const auto d = hnsw::load_fvecs(p.string());
    EXPECT_EQ(d.dim, 3u);
    EXPECT_EQ(d.count, 2u);
    EXPECT_EQ(d.row(0)[0], 1.f);
    EXPECT_EQ(d.row(1)[2], 6.f);
    fs::remove(p);
}

TEST(Dataset, LoadsIvecs) {
    const auto p = tmp_path("ok.ivecs");
    write_vecs<int32_t>(p, {{7, 8}, {9, 10}, {11, 12}});
    const auto d = hnsw::load_ivecs(p.string());
    EXPECT_EQ(d.dim, 2u);
    EXPECT_EQ(d.count, 3u);
    EXPECT_EQ(d.row(2)[1], 12);
    fs::remove(p);
}

TEST(Dataset, MissingFileThrows) {
    EXPECT_THROW(hnsw::load_fvecs("/nonexistent/nope.fvecs"), std::runtime_error);
}

TEST(Dataset, EmptyFileThrows) {
    const auto p = tmp_path("empty.fvecs");
    std::ofstream(p).close();
    EXPECT_NE(error_of(p).find("empty"), std::string::npos);
    fs::remove(p);
}

TEST(Dataset, TruncatedFileThrows) {
    const auto p = tmp_path("trunc.fvecs");
    write_vecs<float>(p, {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    fs::resize_file(p, fs::file_size(p) - 4);
    EXPECT_NE(error_of(p).find("truncated"), std::string::npos);
    fs::remove(p);
}

TEST(Dataset, InvalidDimensionThrows) {
    const auto p = tmp_path("baddim.fvecs");
    write_vecs<float>(p, {{1.f, 2.f}});
    patch_int32(p, 0, -5);
    EXPECT_NE(error_of(p).find("invalid dimension"), std::string::npos);
    fs::remove(p);
}

TEST(Dataset, InconsistentDimensionThrows) {
    const auto p = tmp_path("mixed.fvecs");
    write_vecs<float>(p, {{1.f, 2.f}, {3.f, 4.f}});
    patch_int32(p, 12, 3);  // second record's header (4 + 8 bytes in)
    EXPECT_NE(error_of(p).find("record 1"), std::string::npos);
    fs::remove(p);
}
