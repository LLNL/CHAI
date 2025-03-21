##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_unit_test_depends
     chai umpire gtest)
     
blt_list_append(TO chai_unit_test_depends ELEMENTS cuda IF ${CHAI_ENABLE_CUDA})
blt_list_append(TO chai_unit_test_depends ELEMENTS blt::hip IF ${CHAI_ENABLE_HIP})

blt_add_executable(
  NAME managed_array_unit_tests
  SOURCES managed_array_unit_tests.cpp
  DEPENDS_ON ${chai_unit_test_depends})

target_include_directories(
  managed_array_unit_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_unit_test
  COMMAND managed_array_unit_tests)

blt_add_executable(
  NAME array_manager_unit_tests
  SOURCES array_manager_unit_tests.cpp
  DEPENDS_ON ${chai_unit_test_depends})

target_include_directories(
  array_manager_unit_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME array_manager_unit_test
  COMMAND array_manager_unit_tests)

if (CHAI_ENABLE_MANAGED_PTR)
  blt_add_executable(
    NAME managed_ptr_unit_tests
    SOURCES managed_ptr_unit_tests.cpp
    DEPENDS_ON ${chai_unit_test_depends})

  target_include_directories(
    managed_ptr_unit_tests
    PUBLIC ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME managed_ptr_unit_test
    COMMAND managed_ptr_unit_tests)
endif ()
