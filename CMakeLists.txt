##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0025 NEW)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

project(Chai LANGUAGES C CXX VERSION 2025.03.1)
cmake_minimum_required(VERSION 3.23)

include(cmake/SetupChaiOptions.cmake)

set(BLT_CXX_STD c++14 CACHE STRING "")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  message(STATUS "Setting CMAKE_CXX_EXTENSIONS to ON for PGI Compiler")
  set( CMAKE_CXX_EXTENSIONS ON )
endif()

message(STATUS "Using CMake version ${CMAKE_VERSION}")

################################
# BLT
################################
if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

#######################################
# Options that depend on BLT Options
#######################################
cmake_dependent_option( CHAI_ENABLE_CUDA "Build CHAI with CUDA support" On
                        "ENABLE_CUDA" Off )
cmake_dependent_option( CHAI_ENABLE_HIP "Build CHAI with HIP" On
                       "ENABLE_HIP" Off )
cmake_dependent_option( CHAI_ENABLE_OPENMP "Build CHAI with OpenMP" On
                       "ENABLE_OPENMP" Off )
cmake_dependent_option( CHAI_ENABLE_MPI "Build CHAI with MPI" On
                       "ENABLE_MPI" Off )

cmake_dependent_option(CHAI_ENABLE_TESTS "Build CHAI tests" On
                       "ENABLE_TESTS" Off)
cmake_dependent_option(CHAI_ENABLE_BENCHMARKS "Build CHAI benchmarks" On
                       "ENABLE_BENCHMARKS" Off)
cmake_dependent_option(CHAI_ENABLE_EXAMPLES "Build CHAI examples" On
                       "ENABLE_EXAMPLES" Off )

cmake_dependent_option(CHAI_ENABLE_DOCS "Build CHAI docs" On
                       "ENABLE_DOCS" Off)
cmake_dependent_option( CHAI_ENABLE_GMOCK "Build CHAI with gmock" On
                        "ENABLE_GMOCK" Off )

include(cmake/ChaiBasics.cmake)

add_subdirectory(src)

if (CHAI_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (CHAI_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (CHAI_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (CHAI_ENABLE_DOCS)
  add_subdirectory(docs)
endif()

if (CHAI_ENABLE_REPRODUCERS)
  add_subdirectory(reproducers)
endif()
