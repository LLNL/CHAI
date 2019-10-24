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

if (ENABLE_OPENMP)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    openmp)
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

if (ENABLE_RAJA_PLUGIN)
  set(raja_test_depends
    ${managed_array_test_depends}
    raja)

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
