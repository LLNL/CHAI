##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0048 NEW)

project(Chai LANGUAGES CXX VERSION 2.3.0)

set(ENABLE_CUDA Off CACHE BOOL "Enable CUDA")
set(ENABLE_HIP Off CACHE BOOL "Enable HIP")
set(ENABLE_GPU_SIMULATION_MODE Off CACHE BOOL "Enable GPU Simulation Mode")
set(ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP")
set(ENABLE_MPI Off CACHE BOOL "Enable MPI (for umpire replay only)")
set(ENABLE_BENCHMARKS On CACHE BOOL "Enable benchmarks")
option(ENABLE_IMPLICIT_CONVERSIONS "Enable implicit conversions to-from raw pointers" On)
option(DISABLE_RM "Make ManagedArray a thin wrapper" Off)
mark_as_advanced(DISABLE_RM)
option(ENABLE_UM "Use CUDA unified (managed) memory" Off)
option(ENABLE_PINNED "Use pinned host memory" Off)
option(ENABLE_RAJA_PLUGIN "Build plugin to set RAJA execution spaces" Off)
option(CHAI_ENABLE_GPU_ERROR_CHECKING "Enable GPU error checking" On)
option(CHAI_ENABLE_MANAGED_PTR "Enable managed_ptr" On)
option(CHAI_DEBUG "Enable Debug Logging.")
set(ENABLE_RAJA_NESTED_TEST ON CACHE BOOL "Enable raja-chai-nested-tests, which fails to build on Debug CUDA builds.")

set(ENABLE_TESTS On CACHE BOOL "")
set(ENABLE_EXAMPLES On CACHE BOOL "")
set(ENABLE_DOCS Off CACHE BOOL "")

# options for Umpire as TPL
set(ENABLE_GMOCK On CACHE BOOL "")
set(ENABLE_ASSERTS "Build Umpire with assert() enabled" On)
set(ENABLE_GTEST_DEATH_TESTS ${ENABLE_ASSERTS} CACHE BOOL "")

if (ENABLE_UM AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "Option ENABLE_UM requires ENABLE_CUDA")
endif()

set(ENABLE_COPY_HEADERS Off CACHE BOOL "")
set(BLT_CXX_STD c++11 CACHE STRING "")

if (ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.9)
else ()
  cmake_minimum_required(VERSION 3.8)
endif()

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

include(cmake/ChaiBasics.cmake)

add_subdirectory(src)

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif ()

if (ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif()
