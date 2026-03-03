//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "gtest/gtest.h"

#include <optional>

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  std::optional<::chai::expt::Context> context = contextManager.getContext();

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::HOST;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    std::optional<::chai::expt::Context> scopeContext = contextManager.getContext();
    ASSERT_TRUE(scopeContext.has_value());
    EXPECT_EQ(scopeContext.value(), tempContext);
  }

  EXPECT_EQ(contextManager.getContext(), context);
}

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, DEVICE) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  std::optional<::chai::expt::Context> context = contextManager.getContext();

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::DEVICE;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    std::optional<::chai::expt::Context> scopeContext = contextManager.getContext();
    ASSERT_TRUE(scopeContext.has_value());
    EXPECT_EQ(scopeContext.value(), tempContext);
  }

  EXPECT_EQ(contextManager.getContext(), context);
}
