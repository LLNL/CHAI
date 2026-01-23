//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ManagedArrayPointer.hpp"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdlib>

  /**
   * Minimal "ManagerType" for exercising ManagedArrayPointer in unit tests.
   *
   * Requirements satisfied (as used by ManagedArrayPointer):
   *  - void resize(std::size_t)
   *  - std::size_t size() const
   *  - ElementType* data()
   *
   * Owns storage on host via std::realloc.
   */
  template <typename ElementType>
  class TestArrayManager
  {
    public:
      TestArrayManager() = default;

      void resize(std::size_t size)
      {
        m_size = size;
        m_data = static_cast<ElementType*>(std::realloc(m_data, size*sizeof(ElementType)));
      }

      std::size_t size() const
      {
        return m_size;
      }

      ElementType* data()
      {
        return m_data;
      }

    private:
      std::size_t m_size{0};
      ElementType* m_data{nullptr};
  };  // class TestArrayManager

template <typename ElementType>
using TestArrayPointer = ::chai::expt::ManagedArrayPointer<ElementType, TestArrayManager<ElementType>>;

template <typename ElementType>
using ConstTestArrayPointer = ::chai::expt::ManagedArrayPointer<const ElementType, TestArrayManager<ElementType>>;

TEST(ManagedArrayPointer, DefaultConstructor) {
  TestArrayPointer<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(ManagedArrayPointer, ManagerConstructor) {
  TestArrayPointer<int> a{TestArrayManager<int>{}};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, CopyConstructor) {
  TestArrayPointer<int> a{TestArrayManager<int>()};
  TestArrayPointer<int> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ManagedArrayPointer, ConvertingConstructor) {
  TestArrayPointer<int> a{TestArrayManager<int>()};
  ConstTestArrayPointer<int> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ManagedArrayPointer, CopyAssignmentOperator) {
  TestArrayPointer<int> a;
  a = TestArrayPointer<int>(TestArrayManager<int>());
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, Resize) {
  const std::size_t n = 10;
  TestArrayPointer<int> a;
  a.resize(n);
  EXPECT_EQ(a.size(), n);
  EXPECT_NE(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, Data) {
  const std::size_t n = 10;
  TestArrayPointer<int> a;
  a.resize(n);
  a.update();

  for (std::size_t i = 0; i < n; ++i)
  {
    a[i] = i;
  }

  int* data = a.data();

  for (std::size_t i = 0; i < n; ++i)
  {
    EXPECT_EQ(data[i], i);
  }

  a.free();
}

TEST(ManagedArrayPointer, LambdaCapture) {
  const std::size_t n = 10;
  TestArrayPointer<int> a;
  a.resize(n);

  auto f = [=] (std::size_t i)
  {
    a[i] = i;
  };

  for (std::size_t i = 0; i < n; ++i)
  {
    f(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < n; ++i)
  {
    EXPECT_EQ(data[i], i);
  }

  a.free();
}
