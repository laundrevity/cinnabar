cmake_minimum_required(VERSION 3.16)
project(cinnabar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core engine: pure C++, no dependencies.
add_library(cinnabar_engine_lib STATIC src/engine.cpp src/encoder.cpp)
target_include_directories(cinnabar_engine_lib PUBLIC include)
target_compile_options(cinnabar_engine_lib PRIVATE -Wall -Wextra)

# Unit tests (ctest).
enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE cinnabar_engine_lib)
add_test(NAME engine_tests COMMAND test_engine)

# Python bindings (optional; -DBUILD_PYTHON=OFF to skip). pybind11 is fetched at
# configure time, so no separate install is needed (requires network + git).
option(BUILD_PYTHON "Build the pybind11 Python module" ON)
if(BUILD_PYTHON)
  # Force pybind11 onto the modern FindPython so it honours -DPython_EXECUTABLE.
  # Without this, pybind11's legacy finder grabs an unrelated system Python (e.g.
  # /usr/bin/python3.9) and builds an unimportable module for the wrong version.
  set(PYBIND11_FINDPYTHON ON)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
  # Module name must match PYBIND11_MODULE(...) in bind.cpp -> `import cinnabar_engine`.
  pybind11_add_module(cinnabar_engine bindings/bind.cpp src/engine.cpp src/encoder.cpp)
  target_include_directories(cinnabar_engine PRIVATE include)
endif()
