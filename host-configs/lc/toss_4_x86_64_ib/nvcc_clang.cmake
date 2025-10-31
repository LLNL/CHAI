##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# Use gcc std libraries
set(GCC_VER "13.3.1" CACHE STRING "")
set(GCC_DIR "/usr/tce/packages/gcc/gcc-${GCC_VER}-magic" CACHE PATH "")

# Use clang toolchain for host code compilers
set(CLANG_VER "14.0.6" CACHE STRING "")
set(CLANG_DIR "/usr/tce/packages/clang/clang-${CLANG_VER}-magic" CACHE PATH "")

set(CMAKE_C_COMPILER "${CLANG_DIR}/bin/clang" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_DIR}" CACHE STRING "")

set(CMAKE_CXX_COMPILER "${CLANG_DIR}/bin/clang++" CACHE PATH "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_DIR}" CACHE STRING "")

# Use nvcc as the device code compiler
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_VER "12.9.1" CACHE STRING "")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-${CUDA_VER}" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=--gcc-toolchain=${GCC_DIR} -Wno-deprecated-gpu-targets -Wno-unused-command-line-argument" CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "")

# Prevent incorrect implicit libraries from being linked in
set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-10.3.1/lib/gcc/x86_64-redhat-linux/10;/usr/tce/packages/gcc/gcc-10.3.1/lib64;/lib64;/usr/lib64;/lib;/usr/lib" CACHE STRING "")

set(UMPIRE_FMT_TARGET "fmt::fmt" CACHE STRING "")
