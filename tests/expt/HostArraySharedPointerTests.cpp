//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/HostArrayManager.hpp"
#include "chai/expt/ManagedArraySharedPointer.hpp"
#include "gtest/gtest.h"
#include <cstddef>
#include <cstdlib>

template <typename ElementType>
using HostArrayManager = ::chai::expt::HostArrayManager<ElementType>;

template <typename ElementType>
using HostArraySharedPointer = ::chai::expt::ManagedArraySharedPointer<ElementType, HostArrayManager<ElementType>>;

template <typename ElementType>
using ConstHostArraySharedPointer = ::chai::expt::ManagedArraySharedPointer<const ElementType, HostArrayManager<ElementType>>;

TEST(HostArraySharedPointer, DefaultConstructor) {
  HostArraySharedPointer<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArraySharedPointer, ManagerDefaultConstructor) {
  HostArraySharedPointer<int> a{HostArrayManager<int>()};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArraySharedPointer, MakeManagerDefaultConstructor) {
  HostArraySharedPointer<int> a = HostArraySharedPointer<int>::make();
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArraySharedPointer, ManagerSizeConstructor) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};
  EXPECT_EQ(a.size(), N);
  ASSERT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

TEST(HostArraySharedPointer, MakeManagerSizeConstructor) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a = HostArraySharedPointer<int>::make(N);
  EXPECT_EQ(a.size(), N);
  ASSERT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * Note: The copy constructor is shallow.
 */
TEST(HostArraySharedPointer, CopyConstructor) {
  // Set up array a
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  // Copy construct b
  HostArraySharedPointer<int> b(a);

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
}

TEST(HostArraySharedPointer, ConvertingConstructor) {
  // Set up array a
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  // Use the converting constructor to construct b
  ConstHostArraySharedPointer<int> b(a);

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
}

TEST(HostArraySharedPointer, CopyAssignmentOperator) {
  HostArraySharedPointer<int> a;

  const std::size_t N = 10;
  a = HostArraySharedPointer<int>(HostArrayManager<int>(N));

  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);
}

TEST(HostArraySharedPointer, ResizeAllocate) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a;
  a.resize(N);

  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);

  // For memory checking tools, use the allocated array.
  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

TEST(HostArraySharedPointer, ResizeDeallocate) {
  std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};

  N = 0;
  a.resize(N);

  EXPECT_EQ(a.size(), N);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArraySharedPointer, ResizeSmaller) {
  const std::size_t old_size = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(old_size)};

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
}

TEST(HostArraySharedPointer, ResizeLarger) {
  const std::size_t old_size = 5;
  HostArraySharedPointer<int> a{HostArrayManager<int>(old_size)};

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
}

TEST(HostArraySharedPointer, Data) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], i);
  }
}

TEST(HostArraySharedPointer, Update) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};
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
}

TEST(HostArraySharedPointer, LambdaCapture) {
  const std::size_t N = 10;
  HostArraySharedPointer<int> a{HostArrayManager<int>(N)};

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
}
