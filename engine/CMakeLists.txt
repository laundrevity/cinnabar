cmake_minimum_required(VERSION 3.16)
project(cinnabar_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core engine: pure C++, no dependencies.
add_library(cinnabar_engine STATIC src/engine.cpp)
target_include_directories(cinnabar_engine PUBLIC include)
target_compile_options(cinnabar_engine PRIVATE -Wall -Wextra)

# Unit tests, runnable via ctest.
enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE cinnabar_engine)
add_test(NAME engine_tests COMMAND test_engine)

# (Later) pybind11 module for the Python adapter will be added here.
