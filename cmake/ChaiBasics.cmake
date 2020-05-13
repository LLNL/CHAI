##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if (ENABLE_HIP)
  #bug in ROCm 2.4 is incorrectly warning about missing overrides
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -Wno-inconsistent-missing-override")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupChaiThirdparty.cmake)
