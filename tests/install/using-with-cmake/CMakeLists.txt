###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
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

find_package(chai CONFIG REQUIRED NO_DEFAULT_PATH PATHS ${chai_DIR} ${CHAI_DIR})

add_executable(using-with-cmake using-with-cmake.cpp)
target_link_libraries(using-with-cmake chai)
