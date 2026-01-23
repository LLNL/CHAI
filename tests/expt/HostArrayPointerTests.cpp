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
using HostArrayManager = ::chai::expt::HostArrayManager<ElementType>;

template <typename ElementType>
using HostArrayPointer = ::chai::expt::ManagedArrayPointer<ElementType, HostArrayManager<ElementType>>;

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

  a.free();
}

TEST(HostArrayPointer, ManagerConstructor) {
  const std::size_t N = 10;
  HostArrayPointer<float> a(new HostArrayManager<float>(N));
  EXPECT_EQ(a.size(), N);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(HostArrayPointer, CopyConstructor) {
  const std::size_t size = 10;
  HostArrayPointer<int> a(new HostArrayManager<int>(size));

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    a.update();

    for (std::size_t i = 0; i < size; ++i)
    {
      a[i] = i;
    }
  }

  HostArrayPointer<int> b{a};

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    b.update();
    EXPECT_EQ(a.size(), b.size());

    const int* a_data = a.data();
    const int* b_data = b.data();

    EXPECT_EQ(a_data, b_data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(a_data[i], i);
      EXPECT_EQ(b_data[i], i);
    }
  }
}
