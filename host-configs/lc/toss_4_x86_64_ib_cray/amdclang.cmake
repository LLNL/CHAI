##############################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(ENABLE_HIP ON CACHE BOOL "Enable Hip")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-6.2.0-magic" CACHE PATH "")
set(CMAKE_HIP_ARCHITECTURES "gfx942:xnack+" CACHE STRING "")
set(AMDGPU_TARGETS "gfx942:xnack+" CACHE STRING "")

set(CMAKE_C_COMPILER "${ROCM_PATH}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/amdclang++" CACHE PATH "")

