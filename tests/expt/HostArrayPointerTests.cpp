//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/HostArrayManager.hpp"
#include "chai/expt/ManagedArrayPointer.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdlib>

template <typename ElementType>
using HostArrayManager = ::chai::expt::HostArrayManager<ElementType>;

template <typename ElementType>
using HostArrayPointer = ::chai::expt::ManagedArrayPointer<ElementType, HostArrayManager<ElementType>>;

template <typename ElementType>
using ConstHostArrayPointer = ::chai::expt::ManagedArrayPointer<const ElementType, HostArrayManager<ElementType>>;

TEST(HostArrayPointer, DefaultConstructor) {
  HostArrayPointer<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArrayPointer, ManagerDefaultConstructor) {
  HostArrayPointer<int> a{HostArrayManager<int>()};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(HostArrayPointer, MakeManagerDefaultConstructor) {
  HostArrayPointer<int> a = HostArrayPointer<int>::make();
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(HostArrayPointer, ManagerSizeConstructor) {
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};
  EXPECT_EQ(a.size(), N);
  ASSERT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  a.free();
}

TEST(HostArrayPointer, MakeManagerSizeConstructor) {
  const std::size_t N = 10;
  HostArrayPointer<int> a = HostArrayPointer<int>::make(N);
  EXPECT_EQ(a.size(), N);
  ASSERT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  a.free();
}

/*
 * Note: The copy constructor is shallow.
 */
TEST(HostArrayPointer, CopyConstructor) {
  // Set up array a
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  // Copy construct b
  HostArrayPointer<int> b(a);

  // Check that a is unchanged
  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(a[i], i);
  }

  // Check that b matches a
  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(b[i], i);
  }

  // Check that the copy is shallow
  for (std::size_t i = 0; i < N; ++i)
  {
    b[i] = -i;
    EXPECT_EQ(a[i], -i);
  }

  // Free memory
  a.free();
}

TEST(HostArrayPointer, ConvertingConstructor) {
  // Set up array a
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  // Use the converting constructor to construct b
  ConstHostArrayPointer<int> b(a);

  // Check that a is unchanged
  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(a[i], i);
  }

  // Check that b matches a
  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(b[i], i);
  }

  // Check that the copy is shallow
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = -i;
    EXPECT_EQ(b[i], -i);
  }

  // Free memory
  a.free();
}

TEST(HostArrayPointer, CopyAssignmentOperator) {
  HostArrayPointer<int> a;

  const std::size_t N = 10;
  a = HostArrayPointer<int>(HostArrayManager<int>(N));

  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);
}

TEST(HostArrayPointer, ResizeAllocate) {
  const std::size_t N = 10;
  HostArrayPointer<int> a;
  a.resize(N);

  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  a.free();
}

TEST(HostArrayPointer, ResizeDeallocate) {
  std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};

  N = 0;
  a.resize(N);

  EXPECT_EQ(a.size(), N);
  EXPECT_EQ(a.data(), nullptr);

  a.free();
}

TEST(HostArrayPointer, ResizeSmaller) {
  const std::size_t old_size = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(old_size)};

  for (std::size_t i = 0; i < old_size; ++i)
  {
    a[i] = i;
  }

  const std::size_t new_size = 5;
  a.resize(new_size);

  EXPECT_EQ(a.size(), new_size);
  EXPECT_NE(a.data(), nullptr);

  for (std::size_t i = 0; i < new_size; ++i)
  {
    EXPECT_EQ(a[i], i);
  }

  a.free();
}

TEST(HostArrayPointer, ResizeLarger) {
  const std::size_t old_size = 5;
  HostArrayPointer<int> a{HostArrayManager<int>(old_size)};

  for (std::size_t i = 0; i < old_size; ++i)
  {
    a[i] = i;
  }

  const std::size_t new_size = 10;
  a.resize(new_size);

  EXPECT_EQ(a.size(), new_size);
  EXPECT_NE(a.data(), nullptr);

  for (std::size_t i = 0; i < old_size; ++i)
  {
    EXPECT_EQ(a[i], i);
  }

  a.free();
}

TEST(HostArrayPointer, Data) {
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], i);
  }

  a.free();
}

TEST(HostArrayPointer, Update) {
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};
  a.update();

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], i);
  }

  a.free();
}

TEST(HostArrayPointer, Free) {
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};
  a.free();
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArrayPointer, LambdaCapture) {
  const std::size_t N = 10;
  HostArrayPointer<int> a{HostArrayManager<int>(N)};

  auto f = [=] (std::size_t i)
  {
    a[i] = i;
  };

  for (std::size_t i = 0; i < N; ++i)
  {
    f(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], i);
  }

  a.free();
}
