//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/CopyHidingArrayManager.hpp"
#include "gtest/gtest.h"

TEST(CopyHidingArrayManager, DefaultConstructor) {
  chai::expt::CopyingHidingArrayManager arrayManager{};
  EXPECT_EQ(arrayManager.size(), 0);
  EXPECT_EQ(arrayManager.data(chai::expt::ExecutionContext::NONE), nullptr);
  EXPECT_EQ(arrayManager.data(chai::expt::ExecutionContext::HOST), nullptr);
  EXPECT_EQ(arrayManager.data(chai::expt::ExecutionContext::DEVICE), nullptr);
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