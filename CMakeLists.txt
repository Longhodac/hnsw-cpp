cmake_minimum_required(VERSION 3.24)
project(hnsw_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HNSW_NATIVE_ARCH "Tune for the host CPU (-mcpu/-march=native)" OFF)
option(HNSW_BUILD_TESTS "Build unit tests" ON)
option(HNSW_BUILD_BENCH "Build microbenchmarks" ON)
option(HNSW_BUILD_PYTHON "Build the pybind11 Python module (needs Python3 and pybind11)" OFF)

# ---- Third-party (test/bench only; never linked into the core library) ----
include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
if(HNSW_BUILD_TESTS)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)
endif()
if(HNSW_BUILD_BENCH)
    FetchContent_Declare(benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(benchmark)
endif()

# ---- Shared flags for our own targets (not third-party) ----
add_library(hnsw_options INTERFACE)
target_compile_options(hnsw_options INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
if(HNSW_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        check_cxx_compiler_flag(-mcpu=native HNSW_HAS_MCPU_NATIVE)
        if(HNSW_HAS_MCPU_NATIVE)
            target_compile_options(hnsw_options INTERFACE -mcpu=native)
        endif()
    else()
        check_cxx_compiler_flag(-march=native HNSW_HAS_MARCH_NATIVE)
        if(HNSW_HAS_MARCH_NATIVE)
            target_compile_options(hnsw_options INTERFACE -march=native)
        endif()
    endif()
endif()

# ---- Core library ----
add_library(hnsw
    src/dataset.cpp
    src/brute_force.cpp
    src/metrics.cpp
    src/memory_region.cpp
    src/hnsw_index.cpp
    src/hnsw_index_io.cpp)
target_include_directories(hnsw PUBLIC include)
target_link_libraries(hnsw PRIVATE hnsw_options)

# ---- Eval CLI ----
find_package(Threads REQUIRED)
add_executable(eval tools/eval.cpp)
target_link_libraries(eval PRIVATE hnsw hnsw_options Threads::Threads)

if(HNSW_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(HNSW_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(HNSW_BUILD_PYTHON)
    add_subdirectory(python)
endif()
