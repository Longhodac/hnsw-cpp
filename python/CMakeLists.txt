# Python module `hnsw_cpp` (pybind11). Configure with the python preset:
#   cmake --preset python && cmake --build --preset python && ctest --preset python
# The preset points Python3_EXECUTABLE at .venv/bin/python, so the module matches that Python.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)

# Prefer the pybind11 installed in that Python, so configuring needs no network. Download it
# only when it is missing.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE HNSW_PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE HNSW_PYBIND11_STATUS)
if(HNSW_PYBIND11_STATUS EQUAL 0)
    set(pybind11_DIR "${HNSW_PYBIND11_DIR}")
endif()
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(hnsw_cpp bindings.cpp)
target_link_libraries(hnsw_cpp PRIVATE hnsw hnsw_options Threads::Threads)
set_target_properties(hnsw_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

if(HNSW_BUILD_TESTS)
    add_test(NAME python_binding
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(python_binding PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        TIMEOUT 300)
endif()
