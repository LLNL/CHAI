############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
option(CHAI_ENABLE_GPU_SIMULATION_MODE "Enable GPU Simulation Mode" Off)
option(CHAI_ENABLE_OPENMP "Enable OpenMP" Off)
option(CHAI_ENABLE_MPI "Enable MPI (for umpire replay only)" Off)
option(CHAI_ENABLE_IMPLICIT_CONVERSIONS "Enable implicit conversions to-from raw pointers" On)

option(CHAI_DISABLE_RM "Make ManagedArray a thin wrapper" Off)
mark_as_advanced(CHAI_DISABLE_RM)

option(CHAI_ENABLE_UM "Use CUDA unified (managed) memory" Off)
option(CHAI_ENABLE_PINNED "Use pinned host memory" Off)
option(CHAI_ENABLE_RAJA_PLUGIN "Build plugin to set RAJA execution spaces" Off)
option(CHAI_ENABLE_GPU_ERROR_CHECKING "Enable GPU error checking" On)
option(CHAI_ENABLE_MANAGED_PTR "Enable managed_ptr" On)
option(CHAI_DEBUG "Enable Debug Logging." Off)
option(CHAI_ENABLE_RAJA_NESTED_TEST "Enable raja-chai-nested-tests, which fails to build on Debug CUDA builds." On)

option(CHAI_ENABLE_TESTS "Enable CHAI tests" On)
option(CHAI_ENABLE_BENCHMARKS "Enable benchmarks" On)
option(CHAI_ENABLE_EXAMPLES "Enable CHAI examples" On)
option(CHAI_ENABLE_REPRODUCERS "Enable CHAI reproducers" Off)
option(CHAI_ENABLE_DOCS "Enable CHAI documentation" Off)

# options for Umpire as TPL
set(ENABLE_GMOCK On CACHE BOOL "")
option(CHAI_ENABLE_ASSERTS "Build Umpire with assert() enabled" On)
set(ENABLE_GTEST_DEATH_TESTS ${CHAI_ENABLE_ASSERTS} CACHE BOOL "")

option(CHAI_ENABLE_COPY_HEADERS "Enable CHAI copy headers" Off)

set(ENABLE_CUDA Off CACHE BOOL "Enable CUDA")

if (CHAI_ENABLE_UM AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "Option CHAI_ENABLE_UM requires ENABLE_CUDA")
endif()
