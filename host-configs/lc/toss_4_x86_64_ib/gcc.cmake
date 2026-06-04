##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# Set up software versions
set(GCC_VERSION "11.2.1" CACHE PATH "")

# Set up compilers
set(COMPILER_BASE "/usr/tce/packages/gcc/gcc-${GCC_VERSION}-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/g++" CACHE PATH "")
