##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# Set up software versions
set(CLANG_VERSION "ibm-14.0.5" CACHE PATH "")
set(CUDA_VERSION "11.8.0" CACHE PATH "")
set(GCC_VERSION "11.2.1" CACHE PATH "")

# Set up compilers
set(COMPILER_BASE "/usr/tce/packages/clang/clang-${CLANG_VERSION}" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/clang++" CACHE PATH "")

# Set up compiler flags
set(GCC_HOME "/usr/tce/packages/gcc/gcc-${GCC_VERSION}" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

# Prevent the wrong libraries from being linked in
set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")

# Set up CUDA
set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-${CUDA_VERSION}" CACHE PATH "Path to CUDA")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=--gcc-toolchain=${GCC_HOME}" CACHE STRING "")
