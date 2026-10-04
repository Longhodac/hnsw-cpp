add_executable(hnsw_tests
    test_dataset.cpp
    test_distance.cpp
    test_brute_force.cpp
    test_metrics.cpp
    test_hnsw_storage.cpp
    test_hnsw.cpp)
find_package(Threads REQUIRED)
target_link_libraries(hnsw_tests PRIVATE hnsw hnsw_options GTest::gtest_main Threads::Threads)
target_compile_definitions(hnsw_tests PRIVATE HNSW_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(hnsw_tests DISCOVERY_MODE PRE_TEST)
