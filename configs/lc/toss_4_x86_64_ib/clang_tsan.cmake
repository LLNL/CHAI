##############################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(COMPILER_BASE "/usr/tce/packages/clang/clang-14.0.6-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/clang++" CACHE PATH "")

set(GCC_HOME "/usr/tce/packages/gcc/gcc-12.1.1-magic" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME} -fsanitize=thread -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME} -fsanitize=thread -g" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread" CACHE STRING "")

set(ENABLE_OPENMP ON CACHE BOOL "")
set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")

