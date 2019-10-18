##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (managed_array_test_depends
  chai umpire gtest)

if (ENABLE_CUDA)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    cuda)
endif ()
if (ENABLE_HIP)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    hip)
endif ()

blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${managed_array_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

if (ENABLE_CUDA)
blt_add_executable(
  NAME managed_array_context_tests
  SOURCES managed_array_context_tests.cpp
  DEPENDS_ON ${managed_array_test_depends})

target_include_directories(
  managed_array_context_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_context_test
  COMMAND managed_array_context_tests)
endif ()
