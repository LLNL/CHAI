//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "gtest/gtest.h"

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::Context context = contextManager.getContext();

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::HOST;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    EXPECT_EQ(contextManager.getContext(), tempContext);
  }

  EXPECT_EQ(contextManager.getContext(), context);
}

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, DEVICE) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::Context context = contextManager.getContext();

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::DEVICE;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    EXPECT_EQ(contextManager.getContext(), tempContext);
  }

  EXPECT_EQ(contextManager.getContext(), context);
}
