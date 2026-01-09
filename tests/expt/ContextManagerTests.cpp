//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextManager.hpp"
#include "gtest/gtest.h"

// Test that getInstance returns the same object at the same place in memory
TEST(ContextManager, SingletonInstance) {
  chai::expt::ContextManager& contextManager1 = chai::expt::ContextManager::getInstance();
  chai::expt::ContextManager& contextManager2 = chai::expt::ContextManager::getInstance();  
  EXPECT_EQ(&contextManager1, &contextManager2);
}

// Test that the default execution context is NONE
TEST(ContextManager, DefaultContext) {
  chai::expt::ContextManager& contextManager = chai::expt::ContextManager::getInstance();
  EXPECT_EQ(contextManager.getContext(), chai::expt::Context::NONE);
}

// Test setting and getting the execution context
TEST(ContextManager, Context) {
  chai::expt::ContextManager& contextManager = chai::expt::ContextManager::getInstance();
  chai::expt::Context context = chai::expt::Context::HOST;
  contextManager.setContext(context);
  EXPECT_EQ(contextManager.getContext(), context);
}