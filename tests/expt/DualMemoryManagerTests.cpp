//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/DualMemoryManager.hpp"
#include "gtest/gtest.h"

namespace {
  template <typename F>
  void for_each(::chai::expt::Context context, F&& function);
}

TEST(DualMemoryManager, DefaultConstructor) {
  ::chai::expt::DualMemoryManager<int> dualMemoryManager;
  EXPECT_EQ(dualMemoryManager.size(), 0);
  EXPECT_EQ(dualMemoryManager.data(), nullptr);

  std::size_t size = 10;
  dualMemoryManager.resize(size);
  EXPECT_EQ(dualMemoryManager.size(), size);
  EXPECT_EQ(dualMemoryManager.data(), nullptr);

  {
    ::chai::expt::ContextGuard contextGuard(::chai::expt::Context::HOST);

    EXPECT_EQ(dualMemoryManager.size(), size);
    EXPECT_NE(dualMemoryManager.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard(::chai::expt::Context::DEVICE);

    EXPECT_EQ(dualMemoryManager.size(), size);
    EXPECT_NE(dualMemoryManager.data(), nullptr);
  }
}