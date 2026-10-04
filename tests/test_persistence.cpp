#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hnsw/hnsw_index.hpp"
#include "hnsw/index_file.hpp"
#include "hnsw_test_access.hpp"

using hnsw::HnswIndex;
using hnsw::HnswTestAccess;
namespace file = hnsw::index_file;
namespace fs = std::filesystem;

namespace {

std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

std::unique_ptr<HnswIndex> make_index(size_t n, size_t dim, size_t M, uint64_t seed,
                                      size_t threads = 1) {
    const auto base = random_vectors(n, dim, static_cast<uint32_t>(seed) + 1);
    auto idx = std::make_unique<HnswIndex>(dim, M, 100, n, seed);
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (size_t i = t; i < n; i += threads) idx->add(&base[i * dim]);
        });
    }
    for (auto& th : pool) th.join();
    return idx;
}

using Results = std::vector<std::vector<hnsw::Neighbor>>;

Results run_queries(const HnswIndex& idx, const std::vector<float>& queries, size_t dim, size_t ef) {
    Results out;
    for (size_t q = 0; q < queries.size() / dim; ++q) out.push_back(idx.search(&queries[q * dim], 10, ef));
    return out;
}

std::vector<char> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void write_file(const std::string& p, const std::vector<char>& bytes) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

file::Header header_of(const std::vector<char>& bytes) {
    file::Header h;
    std::memcpy(&h, bytes.data(), sizeof h);
    return h;
}

// Writes the header back, with a valid checksum unless the test wants a damaged one.
void put_header(std::vector<char>& bytes, file::Header h, bool fix_crc = true) {
    if (fix_crc) h.header_crc = file::header_crc(h);
    std::memcpy(bytes.data(), &h, sizeof h);
}

class Persistence : public ::testing::Test {
protected:
    std::string path(const std::string& tag = "main") {
        const std::string name =
            std::string("hnsw_") + ::testing::UnitTest::GetInstance()->current_test_info()->name() +
            "_" + tag + "_" + std::to_string(::getpid()) + ".idx";
        const std::string full = (fs::path(::testing::TempDir()) / name).string();
        paths_.push_back(full);
        return full;
    }
    void TearDown() override {
        for (const auto& p : paths_) {
            std::error_code ec;
            fs::remove(p, ec);
            fs::remove(p + ".tmp", ec);
        }
    }

private:
    std::vector<std::string> paths_;
};

}  // namespace

TEST_F(Persistence, SavedFileHasTheDocumentedLayout) {
    const auto idx = make_index(1500, 16, 12, 3);
    const std::string p = path();
    idx->save(p);

    const auto bytes = read_file(p);
    ASSERT_GE(bytes.size(), sizeof(file::Header));
    const file::Header h = header_of(bytes);
    EXPECT_EQ(file::validate(h, bytes.size()), "");
    EXPECT_EQ(h.count, 1500u);
    EXPECT_EQ(h.dim, 16u);
    EXPECT_EQ(h.M, 12u);
    EXPECT_EQ(h.version, file::kVersion);
    EXPECT_NE(h.entry_id, file::kNoEntry);
    EXPECT_GE(h.arena_slots, 1u);

    const file::Layout l = file::compute_layout(h.count, h.dim, h.M, h.arena_slots);
    EXPECT_EQ(bytes.size(), l.file_size);
    for (const uint64_t off : {l.vectors_off, l.levels_off, l.level0_off, l.upper_offset_off, l.arena_off}) {
        EXPECT_EQ(off % file::kAlign, 0u);
    }
    EXPECT_FALSE(fs::exists(p + ".tmp"));  // the temporary file was renamed away
}

TEST_F(Persistence, LoadedIndexReturnsIdenticalResults) {
    const size_t dim = 16;
    const auto idx = make_index(2000, dim, 12, 5);
    const std::string p = path();
    idx->save(p);
    const auto loaded = HnswIndex::load(p);

    EXPECT_TRUE(loaded->read_only());
    EXPECT_EQ(loaded->size(), idx->size());
    EXPECT_EQ(loaded->dim(), idx->dim());
    EXPECT_EQ(loaded->name(), idx->name());
    const auto queries = random_vectors(200, dim, 99);
    for (const size_t ef : {10u, 50u, 200u}) {
        EXPECT_EQ(run_queries(*loaded, queries, dim, ef), run_queries(*idx, queries, dim, ef))
            << "ef " << ef;
    }
    HnswTestAccess::validate(*loaded);
    EXPECT_NO_THROW(HnswIndex::load(p, HnswIndex::LoadOptions{.verify = true}));
}

TEST_F(Persistence, ParallelBuiltIndexRoundTrips) {
    const size_t dim = 8;
    const auto idx = make_index(1500, dim, 8, 7, 4);
    const std::string p = path();
    idx->save(p);
    const auto loaded = HnswIndex::load(p, HnswIndex::LoadOptions{.verify = true});
    const auto queries = random_vectors(100, dim, 98);
    EXPECT_EQ(run_queries(*loaded, queries, dim, 60), run_queries(*idx, queries, dim, 60));
}

TEST_F(Persistence, EmptyIndexRoundTrips) {
    HnswIndex idx(4, 8, 50, 100, 1);
    const std::string p = path();
    idx.save(p);
    const auto loaded = HnswIndex::load(p, HnswIndex::LoadOptions{.verify = true});
    EXPECT_EQ(loaded->size(), 0u);
    const float q[4] = {0, 0, 0, 0};
    EXPECT_TRUE(loaded->search(q, 5, 10).empty());
}

TEST_F(Persistence, SingleNodeRoundTrips) {
    HnswIndex idx(2, 4, 20, 10, 1);
    const float v[2] = {3, 4};
    idx.add(v);
    const std::string p = path();
    idx.save(p);
    const auto loaded = HnswIndex::load(p);
    const auto r = loaded->search(v, 3, 10);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].second, 0u);
    EXPECT_FLOAT_EQ(r[0].first, 0.f);
}

TEST_F(Persistence, LoadedIndexIsReadOnly) {
    const auto idx = make_index(100, 4, 6, 1);
    const std::string p = path();
    idx->save(p);
    const auto loaded = HnswIndex::load(p);
    const float v[4] = {0, 0, 0, 0};
    EXPECT_THROW(loaded->add(v), std::logic_error);
    EXPECT_EQ(loaded->size(), 100u);
}

TEST_F(Persistence, LoadedIndexServesParallelSearches) {
    const size_t dim = 16;
    const auto idx = make_index(2000, dim, 12, 11);
    const std::string p = path();
    idx->save(p);
    const auto loaded = HnswIndex::load(p);
    const auto queries = random_vectors(150, dim, 97);
    const Results want = run_queries(*idx, queries, dim, 60);

    std::vector<Results> got(4);
    std::vector<std::thread> pool;
    for (size_t t = 0; t < got.size(); ++t) {
        pool.emplace_back([&, t] {
            for (int round = 0; round < 3; ++round) got[t] = run_queries(*loaded, queries, dim, 60);
        });
    }
    for (auto& th : pool) th.join();
    for (size_t t = 0; t < got.size(); ++t) EXPECT_EQ(got[t], want) << "thread " << t;
}

TEST_F(Persistence, SaveReplacesAnExistingFile) {
    const std::string p = path();
    make_index(100, 4, 6, 1)->save(p);
    make_index(300, 4, 6, 2)->save(p);
    EXPECT_EQ(HnswIndex::load(p)->size(), 300u);
    EXPECT_FALSE(fs::exists(p + ".tmp"));
}

TEST_F(Persistence, SaveToMissingDirectoryThrowsAndLeavesNothing) {
    const auto idx = make_index(50, 4, 6, 1);
    const std::string p = (fs::path(::testing::TempDir()) / "hnsw_no_such_dir" / "x.idx").string();
    EXPECT_THROW(idx->save(p), std::runtime_error);
    EXPECT_FALSE(fs::exists(p));
    EXPECT_FALSE(fs::exists(p + ".tmp"));
}

TEST_F(Persistence, LoadingAMissingFileThrows) {
    EXPECT_THROW(HnswIndex::load(path() + ".nope"), std::runtime_error);
}

// Each header or size problem must be refused with a message that names it.
TEST_F(Persistence, DamagedFilesAreRejectedWithAClearMessage) {
    const auto idx = make_index(800, 8, 8, 13);
    const std::string good = path("good");
    idx->save(good);
    const std::vector<char> original = read_file(good);

    struct Case {
        const char* name;
        std::function<void(std::vector<char>&)> damage;
        const char* expect;
    };
    const std::vector<Case> cases = {
        {"bad magic", [](auto& b) { b[0] = 'X'; }, "bad magic"},
        {"not an index at all", [](auto& b) { b.assign(300, 'a'); }, "bad magic"},
        {"future version", [](auto& b) { auto h = header_of(b); h.version = 2; put_header(b, h); },
         "unsupported format version 2"},
        {"other byte order", [](auto& b) { auto h = header_of(b); h.endian_marker = 0x04030201; put_header(b, h); },
         "byte order"},
        {"flipped header bit", [](auto& b) { b[offsetof(file::Header, dim)] ^= 1; }, "checksum"},
        {"reserved field set", [](auto& b) { auto h = header_of(b); h.reserved = 1; put_header(b, h); },
         "reserved"},
        {"zero dimension", [](auto& b) { auto h = header_of(b); h.dim = 0; put_header(b, h); },
         "dimension"},
        {"entry out of range", [](auto& b) { auto h = header_of(b); h.entry_id = 100000; put_header(b, h); },
         "entry node out of range"},
        {"inflated count", [](auto& b) { auto h = header_of(b); h.count += 1000; put_header(b, h); },
         "section offsets"},
        {"truncated to half", [](auto& b) { b.resize(b.size() / 2); }, "truncated or extended"},
        {"extended", [](auto& b) { b.insert(b.end(), 4096, '\0'); }, "truncated or extended"},
        {"shorter than a header", [](auto& b) { b.resize(40); }, "too small"},
        {"empty", [](auto& b) { b.clear(); }, "too small"},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        std::vector<char> bytes = original;
        c.damage(bytes);
        const std::string p = path(std::string("bad") + std::to_string(&c - cases.data()));
        write_file(p, bytes);
        try {
            HnswIndex::load(p);
            ADD_FAILURE() << "load accepted a damaged file";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(c.expect), std::string::npos)
                << "message was: " << e.what();
        }
    }
}

// The cheap load checks do not read links. Damage inside a section loads, and only the
// opt-in verify pass finds it. This is why load() is documented as for files you wrote.
TEST_F(Persistence, VerifyCatchesDamagedLinksThatAHeaderCheckCannotSee) {
    const size_t dim = 8, n = 1200;
    const auto idx = make_index(n, dim, 8, 17);
    const std::string good = path("good");
    idx->save(good);
    const std::vector<char> original = read_file(good);
    const file::Header h = header_of(original);
    const size_t stride0 = 1 + 2 * h.M;

    auto u32_at = [](std::vector<char>& b, uint64_t off) { return reinterpret_cast<uint32_t*>(b.data() + off); };
    auto first_linked_node = [&](std::vector<char>& b) {
        for (uint32_t id = 0; id < h.count; ++id) {
            if (u32_at(b, h.level0_off)[id * stride0] > 0) return id;
        }
        return UINT32_MAX;
    };
    auto first_upper_node = [&](std::vector<char>& b) {
        for (uint32_t id = 0; id < h.count; ++id) {
            if (static_cast<uint8_t>(b[h.levels_off + id]) > 0) return id;
        }
        return UINT32_MAX;
    };

    struct Case {
        const char* name;
        std::function<void(std::vector<char>&)> damage;
        const char* expect;
    };
    const std::vector<Case> cases = {
        {"link to a missing node",
         [&](auto& b) { u32_at(b, h.level0_off)[first_linked_node(b) * stride0 + 1] = 5000000; },
         "does not exist"},
        {"link to itself",
         [&](auto& b) { const auto id = first_linked_node(b); u32_at(b, h.level0_off)[id * stride0 + 1] = id; },
         "link to itself"},
        {"neighbor count above the limit",
         [&](auto& b) { u32_at(b, h.level0_off)[first_linked_node(b) * stride0] = 1000; },
         "above the limit"},
        {"upper offset outside the arena",
         [&](auto& b) { u32_at(b, h.upper_offset_off)[first_upper_node(b)] = 4000000000u; },
         "outside the arena"},
        {"level-0 node with an upper offset",
         [&](auto& b) {
             for (uint32_t id = 0; id < h.count; ++id) {
                 if (b[h.levels_off + id] == 0) { u32_at(b, h.upper_offset_off)[id] = 1; return; }
             }
         },
         "level-0 node has an upper offset"},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        std::vector<char> bytes = original;
        c.damage(bytes);
        const std::string p = path(std::string("dmg") + std::to_string(&c - cases.data()));
        write_file(p, bytes);
        EXPECT_NO_THROW(HnswIndex::load(p)) << "the header is intact, so the cheap load succeeds";
        try {
            HnswIndex::load(p, HnswIndex::LoadOptions{.verify = true});
            ADD_FAILURE() << "verify accepted the damage";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(c.expect), std::string::npos)
                << "message was: " << e.what();
        }
    }
}

TEST_F(Persistence, EntryLevelMismatchIsCaughtByTheCheapLoad) {
    const auto idx = make_index(500, 8, 8, 19);
    const std::string good = path("good");
    idx->save(good);
    std::vector<char> bytes = read_file(good);
    const file::Header h = header_of(bytes);
    bytes[h.levels_off + h.entry_id] = static_cast<char>(h.entry_level + 1);
    const std::string p = path("bad");
    write_file(p, bytes);
    try {
        HnswIndex::load(p);
        ADD_FAILURE() << "load accepted a mismatched entry level";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("entry node"), std::string::npos) << e.what();
    }
}
