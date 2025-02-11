###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and CHAI project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_HIP)
  cmake_minimum_required(VERSION 3.23)
else()
  cmake_minimum_required(VERSION 3.20)
endif()

project(using_with_cmake)

if (ENABLE_CUDA)
  enable_language(CUDA)
endif()

if (ENABLE_HIP)
  if (NOT ROCM_PATH)
    find_path(ROCM_PATH
      hip
      ENV{ROCM_DIR}
      ENV{ROCM_PATH}
      ENV{HIP_PATH}
      ${HIP_PATH}/..
      ${HIP_ROOT_DIR}/../
      ${ROCM_ROOT_DIR}
      /opt/rocm)
  endif()

  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${ROCM_PATH}")
  find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH})
endif()

find_package(chai CONFIG REQUIRED NO_DEFAULT_PATH PATHS ${chai_DIR} ${CHAI_DIR})

# Remove implicitly added link directories added by CMake that are problematic when
# the default system libraries are older than the ones used by the compiler
if (BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE)
  list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_DIRECTORIES
                   ${BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
  list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES
                   ${BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
  list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES
                   ${BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE})
endif ()

add_executable(using-with-cmake using-with-cmake.cpp)
target_link_libraries(using-with-cmake chai)
