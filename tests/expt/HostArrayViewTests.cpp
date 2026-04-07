//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/HostArrayManager.hpp"
#include "chai/expt/ManagedArrayView.hpp"
#include "gtest/gtest.h"

#include <cstddef>

template <typename ElementType>
using HostArrayManager = ::chai::expt::HostArrayManager<ElementType>;

template <typename ElementType>
using HostArrayView = ::chai::expt::ManagedArrayView<ElementType, HostArrayManager<ElementType>>;

template <typename ElementType>
using ConstHostArrayView = ::chai::expt::ManagedArrayView<const ElementType, HostArrayManager<ElementType>>;

TEST(HostArrayView, DefaultConstructor) {
  HostArrayView<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArrayView, ManagerDefaultConstructor) {
  HostArrayManager<int> manager;
  HostArrayView<int> a{manager};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(HostArrayView, ManagerSizeConstructor) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};
  EXPECT_EQ(a.size(), N);
  ASSERT_NE(a.data(), nullptr);

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<int>(i);
  }
}

TEST(HostArrayView, CopyConstructor) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<int>(i);
  }

  HostArrayView<int> b(a);

  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(b[i], static_cast<int>(i));
  }

  for (std::size_t i = 0; i < N; ++i)
  {
    b[i] = -static_cast<int>(i);
    EXPECT_EQ(a[i], -static_cast<int>(i));
  }
}

TEST(HostArrayView, ConvertingConstructor) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<int>(i);
  }

  ConstHostArrayView<int> b(a);

  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(b[i], static_cast<int>(i));
  }
}

TEST(HostArrayView, CopyAssignmentOperator) {
  HostArrayView<int> a;

  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  a = HostArrayView<int>(manager);

  EXPECT_EQ(a.size(), N);
  EXPECT_NE(a.data(), nullptr);
}

TEST(HostArrayView, Data) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<int>(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
}

TEST(HostArrayView, Update) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};
  a.update();

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = static_cast<int>(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
}

TEST(HostArrayView, ReflectsManagerResizeAfterUpdate) {
  HostArrayManager<int> manager{4};
  HostArrayView<int> a{manager};

  EXPECT_EQ(a.size(), 4);

  manager.resize(7);
  a.update();

  EXPECT_EQ(a.size(), 7);
  EXPECT_EQ(a.data(), manager.data());
}

TEST(HostArrayView, LambdaCapture) {
  const std::size_t N = 10;
  HostArrayManager<int> manager{N};
  HostArrayView<int> a{manager};

  auto f = [=] (std::size_t i)
  {
    a[i] = static_cast<int>(i);
  };

  for (std::size_t i = 0; i < N; ++i)
  {
    f(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
}
