##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (managed_array_test_depends
  chai umpire gtest)

set (execution_space_test_depends
  chai umpire gtest)


set (array_manager_test_depends
  chai umpire gtest)

if (ENABLE_CUDA)
  set (execution_space_test_depends
    ${execution_space_test_depends}
    cuda)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    cuda)
  set (array_manager_test_depends
    ${array_manager_test_depends}
    cuda)
endif ()
if (ENABLE_HIP)
  set (execution_space_test_depends
    ${execution_space_test_depends}
    hip)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    hip)
  set (array_manager_test_depends
    ${array_manager_test_depends}
    hip)
endif ()
set (chai_unit_test_depends
     chai umpire gtest)
     
blt_list_append(TO chai_unit_test_depends ELEMENTS cuda IF ${CHAI_ENABLE_CUDA})
blt_list_append(TO chai_unit_test_depends ELEMENTS blt::hip IF ${CHAI_ENABLE_HIP})

# ExecutionSpace tests
blt_add_executable(
  NAME execution_space_unit_test
  SOURCES execution_space_unit_tests.cpp
  DEPENDS_ON ${execution_space_test_depends})

target_include_directories(
  execution_space_unit_test
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME execution_space_unit_test
  COMMAND execution_space_unit_test)

# ManagedArray tests
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
