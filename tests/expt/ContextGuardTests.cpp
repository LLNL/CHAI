//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "chai/expt/ContextManager.hpp"
#include "gtest/gtest.h"

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  ::chai::expt::OptionalExecutionContext saved = contextManager.getContext();

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::HostContext{}};
    EXPECT_TRUE(contextManager.hasContext());
  }

  EXPECT_EQ(contextManager.getContext(), saved);
}

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
#if defined(CHAI_ENABLE_CUDA)
TEST(ContextGuard, CUDA) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  ::chai::expt::OptionalExecutionContext saved = contextManager.getContext();

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::CudaContext{}};
    EXPECT_TRUE(contextManager.hasContext());
  }

  EXPECT_EQ(contextManager.getContext(), saved);
}
#endif

#if defined(CHAI_ENABLE_HIP)
TEST(ContextGuard, HIP) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  ::chai::expt::OptionalExecutionContext saved = contextManager.getContext();

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::HipContext{}};
    EXPECT_TRUE(contextManager.hasContext());
  }

  EXPECT_EQ(contextManager.getContext(), saved);
}
#endif
