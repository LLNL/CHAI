############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
option(CHAI_ENABLE_HIP Off CACHE BOOL "Enable HIP")
option(CHAI_ENABLE_GPU_SIMULATION_MODE Off CACHE BOOL "Enable GPU Simulation Mode")
option(CHAI_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP")
option(CHAI_ENABLE_MPI Off CACHE BOOL "Enable MPI (for umpire replay only)")
option(CHAI_ENABLE_IMPLICIT_CONVERSIONS "Enable implicit conversions to-from raw pointers" On)

option(CHAI_DISABLE_RM "Make ManagedArray a thin wrapper" Off)
mark_as_advanced(CHAI_DISABLE_RM)

option(CHAI_ENABLE_UM "Use CUDA unified (managed) memory" Off)
option(CHAI_ENABLE_PINNED "Use pinned host memory" Off)
option(CHAI_ENABLE_RAJA_PLUGIN "Build plugin to set RAJA execution spaces" Off)
option(CHAI_ENABLE_GPU_ERROR_CHECKING "Enable GPU error checking" On)
option(CHAI_ENABLE_MANAGED_PTR "Enable managed_ptr" On)
option(CHAI_DEBUG "Enable Debug Logging.")
option(CHAI_ENABLE_RAJA_NESTED_TEST ON CACHE BOOL "Enable raja-chai-nested-tests, which fails to build on Debug CUDA builds.")

option(CHAI_ENABLE_TESTS On CACHE BOOL "")
option(CHAI_ENABLE_BENCHMARKS On CACHE BOOL "Enable benchmarks")
option(CHAI_ENABLE_EXAMPLES On CACHE BOOL "")
option(CHAI_ENABLE_REPRODUCERS Off CACHE BOOL "")
option(CHAI_ENABLE_DOCS Off CACHE BOOL "")

# options for Umpire as TPL
set(ENABLE_GMOCK On CACHE BOOL "")
option(CHAI_ENABLE_ASSERTS "Build Umpire with assert() enabled" On)
set(ENABLE_GTEST_DEATH_TESTS ${CHAI_ENABLE_ASSERTS} CACHE BOOL "")

option(CHAI_ENABLE_COPY_HEADERS Off CACHE BOOL "")

set(ENABLE_CUDA Off CACHE BOOL "Enable CUDA")

if (CHAI_ENABLE_UM AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "Option CHAI_ENABLE_UM requires ENABLE_CUDA")
endif()
