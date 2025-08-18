//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ExecutionContextManager.hpp"
#include "gtest/gtest.h"

// Test that getInstance returns the same object at the same place in memory
TEST(ExecutionContextManager, SingletonInstance) {
  chai::expt::ExecutionContextManager& executionContextManager1 = chai::expt::ExecutionContextManager::getInstance();
  chai::expt::ExecutionContextManager& executionContextManager2 = chai::expt::ExecutionContextManager::getInstance();  
  EXPECT_EQ(&executionContextManager1, &executionContextManager2);
}

// Test that the default execution context is NONE
TEST(ExecutionContextManager, DefaultExecutionContext) {
  chai::expt::ExecutionContextManager& executionContextManager = chai::expt::ExecutionContextManager::getInstance();
  EXPECT_EQ(executionContextManager.getExecutionContext(), chai::expt::ExecutionContext::NONE);
}

// Test setting and getting the execution context
TEST(ExecutionContextManager, ExecutionContext) {
  chai::expt::ExecutionContextManager& executionContextManager = chai::expt::ExecutionContextManager::getInstance();
  chai::expt::ExecutionContext executionContext = chai::expt::ExecutionContext::HOST;
  executionContextManager.setExecutionContext(executionContext);
  EXPECT_EQ(executionContextManager.getExecutionContext(), executionContext);
}