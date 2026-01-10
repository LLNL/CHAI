##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

blt_add_executable(
  NAME ContextManagerTests
  SOURCES ContextManagerTests.cpp
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON chai gtest)

blt_add_test(
  NAME ContextManagerTests
  COMMAND ContextManagerTests)

blt_add_executable(
  NAME ContextGuardTests
  SOURCES ContextGuardTests.cpp
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON chai gtest)

blt_add_test(
  NAME ContextGuardTests
  COMMAND ContextGuardTests)

if(CHAI_ENABLE_EXPERIMENTAL_RAJA_PLUGIN)
  blt_add_executable(
    NAME ContextRAJAPluginTests
    SOURCES ContextRAJAPluginTests.cpp
    INCLUDES ${PROJECT_BINARY_DIR}/include
    DEPENDS_ON chai gtest)

  blt_add_test(
    NAME ContextRAJAPluginTests
    COMMAND ContextRAJAPluginTests)
endif()
