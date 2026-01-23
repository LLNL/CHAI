//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/HostArrayManager.hpp"
#include "chai/expt/ManagedArrayPointer.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdlib>

template <typename ElementType>
using HostArrayPointer = ::chai::expt::ManagedArrayPointer<ElementType, ::chai::expt::HostArrayManager<ElementType>>;

TEST(HostArrayPointer, DefaultConstructor) {
  HostArrayPointer<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);

  const std::size_t N = 10;
  a.resize(10);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    a.update();
    EXPECT_EQ(a.size(), N);

    int* data = a.data();
    EXPECT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = i;
    }
  }
}
