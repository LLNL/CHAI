##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# Set up software versions
set(ROCM_VERSION "6.4.3" CACHE PATH "")
set(GCC_VERSION "13.3.1" CACHE PATH "")

# Set up compilers
set(COMPILER_BASE "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VERSION}-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/amdclang++" CACHE PATH "")

# Set up compiler flags
set(GCC_HOME "/usr/tce/packages/gcc/gcc-${GCC_VERSION}" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

# Set up HIP
set(ENABLE_HIP ON CACHE BOOL "")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VERSION}-magic" CACHE PATH "")
set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "")
set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "")
