##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_integration_test_depends
     chai umpire gtest)

blt_list_append(TO chai_integration_test_depends ELEMENTS cuda IF ${ENABLE_CUDA})
blt_list_append(TO chai_integration_test_depends ELEMENTS hip IF ${ENABLE_HIP})
blt_list_append(TO chai_integration_test_depends ELEMENTS openmp IF ${ENABLE_OPENMP})

# ManagedArray tests
blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

blt_add_executable(
  NAME managed_ptr_tests
  SOURCES managed_ptr_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_ptr_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_ptr_test
  COMMAND managed_ptr_tests)
  
if (ENABLE_RAJA_PLUGIN)
  set(raja_test_depends
    ${chai_integration_test_depends}
    RAJA)

  blt_add_executable(
    NAME raja-chai-tests 
    SOURCES raja-chai-tests.cpp
    DEPENDS_ON ${raja_test_depends})

  blt_add_test(
    NAME raja-chai-tests
    COMMAND raja-chai-tests)

  target_include_directories(
    raja-chai-tests
    PUBLIC ${PROJECT_BINARY_DIR}/include)

  blt_add_executable(
    NAME raja-chai-nested-tests 
    SOURCES raja-chai-nested.cpp
    DEPENDS_ON ${raja_test_depends})

  blt_add_test(
    NAME raja-chai-nested-tests
    COMMAND raja-chai-nested-tests)

  target_include_directories(
    raja-chai-nested-tests
    PUBLIC ${PROJECT_BINARY_DIR}/include)
endif ()
