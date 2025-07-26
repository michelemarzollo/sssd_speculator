# Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(sssd_speculator LANGUAGES C CXX)

# ——— C++ settings ———
set(CMAKE_CXX_STANDARD      14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS    OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ——— Python & pybind11 ———
# try 3.9, 3.10, 3.11 in turn
find_package(Python 3.9 EXACT COMPONENTS Interpreter Development QUIET)
if (NOT Python_FOUND)
  find_package(Python 3.10 EXACT COMPONENTS Interpreter Development QUIET)
endif()
if (NOT Python_FOUND)
  find_package(Python 3.11 EXACT COMPONENTS Interpreter Development QUIET)
endif()
if (NOT Python_FOUND)
  message(FATAL_ERROR "Python ≥ 3.9 (≤ 3.11) is required")
endif()

# Make sure you pass the right pybind11 dir (see setup.py)
set(pybind11_DIR ${PYBIND11_INCLUDE_DIR})
find_package(pybind11 REQUIRED)


# ——— FetchContent for external libs ———
include(FetchContent)

# libsais (single-file C project)
FetchContent_Declare(
  libsais
  GIT_REPOSITORY https://github.com/IlyaGrebnov/libsais
  GIT_TAG        master
)
FetchContent_MakeAvailable(libsais)

# build libsais as a static library
add_library(libsais_static STATIC
  ${libsais_SOURCE_DIR}/src/libsais.c
)
target_include_directories(libsais_static PUBLIC
  ${libsais_SOURCE_DIR}/include
)

# include spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.11.0
)
# disable examples/tests to speed it up
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# ——— Include source files ———
set(SOURCES
  src/writer.cpp
  src/reader.cpp
  src/trie.cpp
  src/prompt_cache.cpp
  src/utils.cpp
  src/memory_pool.cpp
  src/sssd_speculator.cpp
)

add_definitions(-DLIBSAIS_STATIC)
pybind11_add_module(sssd_speculator ${SOURCES})

target_include_directories(sssd_speculator PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(sssd_speculator PRIVATE
  libsais_static
  spdlog::spdlog
)