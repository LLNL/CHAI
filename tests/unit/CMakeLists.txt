##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (managed_array_test_depends
  chai umpire gtest)

set (array_manager_test_depends
  chai umpire gtest)

if (ENABLE_CUDA)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    cuda)
  set (array_manager_test_depends
    ${array_manager_test_depends}
    cuda)
endif ()
if (ENABLE_HIP)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    hip)
  set (array_manager_test_depends
    ${array_manager_test_depends}
    hip)
endif ()

blt_add_executable(
  NAME managed_array_unit_tests
  SOURCES managed_array_unit_tests.cpp
  DEPENDS_ON ${managed_array_test_depends})

target_include_directories(
  managed_array_unit_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_unit_test
  COMMAND managed_array_unit_tests)

# ArrayManager tests
blt_add_executable(
  NAME array_manager_unit_tests
  SOURCES array_manager_unit_tests.cpp
  DEPENDS_ON ${array_manager_test_depends})

target_include_directories(
  array_manager_unit_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME array_manager_unit_test
  COMMAND array_manager_unit_tests)

