##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(chai_expt_test_headers
    TestHelpers.hpp)

set(chai_expt_test_depends
    chai
    gtest)

if(ENABLE_CUDA)
  set(chai_expt_test_depends
      ${chai_expt_test_depends}
      cuda)
endif()

if(ENABLE_HIP)
  set(chai_expt_test_depends
      ${chai_expt_test_depends}
      blt::hip)
endif()

blt_add_executable(
  NAME ContextManagerTests
  SOURCES ContextManagerTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME ContextManagerTests
  COMMAND ContextManagerTests)

blt_add_executable(
  NAME ContextGuardTests
  SOURCES ContextGuardTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME ContextGuardTests
  COMMAND ContextGuardTests)

if(CHAI_ENABLE_EXPERIMENTAL_RAJA_PLUGIN)
  blt_add_executable(
    NAME ContextRAJAPluginTests
    SOURCES ContextRAJAPluginTests.cpp
    HEADERS ${chai_expt_test_headers}
    INCLUDES ${PROJECT_BINARY_DIR}/include
    DEPENDS_ON ${chai_expt_test_depends})

  blt_add_test(
    NAME ContextRAJAPluginTests
    COMMAND ContextRAJAPluginTests)
endif()
