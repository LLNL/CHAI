//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "gtest/gtest.h"

// Test that getInstance returns the same object at the same place in memory
TEST(ContextManager, SingletonInstance) {
  ::chai::expt::ContextManager& contextManager1 = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::ContextManager& contextManager2 = ::chai::expt::ContextManager::getInstance();
  EXPECT_EQ(&contextManager1, &contextManager2);
}

// Test that the default context is unset
TEST(ContextManager, DefaultContext) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  EXPECT_FALSE(contextManager.hasContext());
}

// Test setting the HOST context
TEST(ContextManager, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  contextManager.setContext(::chai::expt::HostContext{});
  EXPECT_TRUE(contextManager.hasContext());
  EXPECT_TRUE(contextManager.isSynchronized());
  contextManager.clearContext();
  EXPECT_FALSE(contextManager.hasContext());
}

// Test setting a CUDA device stream context
#if defined(CHAI_ENABLE_CUDA)
TEST(ContextManager, CUDA) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  cudaStream_t stream = 0;
  contextManager.setContext(::chai::expt::CudaContext{stream});
  EXPECT_TRUE(contextManager.hasContext());
  EXPECT_TRUE(contextManager.isSynchronized());
  contextManager.markStreamUnsynchronized(stream);
  EXPECT_FALSE(contextManager.isSynchronized());
  contextManager.synchronize();
  EXPECT_TRUE(contextManager.isSynchronized());
  contextManager.clearContext();
}
#endif

// Test setting a HIP device stream context
#if defined(CHAI_ENABLE_HIP)
TEST(ContextManager, HIP) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  hipStream_t stream = 0;
  contextManager.setContext(::chai::expt::HipContext{stream});
  EXPECT_TRUE(contextManager.hasContext());
  EXPECT_TRUE(contextManager.isSynchronized());
  contextManager.markStreamUnsynchronized(stream);
  EXPECT_FALSE(contextManager.isSynchronized());
  contextManager.synchronize();
  EXPECT_TRUE(contextManager.isSynchronized());
  contextManager.clearContext();
}
#endif
