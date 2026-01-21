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

namespace
{
  /**
   * Minimal "ManagerType" for exercising ManagedArrayPointer in unit tests.
   *
   * Requirements satisfied (as used by ManagedArrayPointer):
   *  - void resize_bytes(std::size_t)
   *  - std::size_t size_bytes() const
   *  - void* data()
   *
   * Owns storage on host via std::vector.
   */
  class TestArrayManager
  {
    public:
      TestArrayManager() = default;

      void resize_bytes(std::size_t bytes)
      {
        m_size_bytes = bytes;
        m_data = std::realloc(m_data, bytes);
      }

      std::size_t size_bytes() const
      {
        return m_size_bytes;
      }

      void* data()
      {
        return m_data;
      }

    private:
      std::size_t m_size_bytes{0};
      void* m_data{nullptr};
  };  // class TestArrayManager
}  // namespace

TEST(ManagedArrayPointer, DefaultConstructor) {
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(ManagedArrayPointer, ManagerConstructor) {
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, CopyConstructor) {
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ManagedArrayPointer, ConvertingConstructor) {
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  ::chai::expt::ManagedArrayPointer<const int, TestArrayManager> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ManagedArrayPointer, CopyAssignmentOperator) {
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a;
  a = ::chai::expt::ManagedArrayPointer<int, TestArrayManager>(new TestArrayManager());
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, resize) {
  const std::size_t n = 10;
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a;
  a.resize(n);
  EXPECT_EQ(a.size(), n);
  EXPECT_NE(a.data(), nullptr);
  a.free();
}

TEST(ManagedArrayPointer, data) {
  const std::size_t n = 10;
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a;
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
}

TEST(ManagedArrayPointer, capture) {
  const std::size_t n = 10;
  ::chai::expt::ManagedArrayPointer<int, TestArrayManager> a;
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
}
