//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ArrayPointer.hpp"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdlib>
#include <vector>

namespace
{
  /**
   * Minimal "ManagerType" for exercising ArrayPointer in unit tests.
   *
   * Requirements satisfied (as used by ArrayPointer):
   *  - void resize(std::size_t)
   *  - std::size_t size() const
   *  - T* data()   (and const overload for convenience)
   *
   * Owns storage on host via std::vector.
   */
  class TestArrayManager
  {
    public:
      TestArrayManager() = default;

      void resize(std::size_t bytes)
      {
        m_size = bytes;
        m_data = std::realloc(m_data, bytes);
      }

      std::size_t size() const
      {
        return m_size;
      }

      void* data()
      {
        return m_data;
      }

    private:
      std::size_t m_size{0};
      void* m_data{nullptr};
  };  // class TestArrayManager
}  // namespace

TEST(ArrayPointer, DefaultConstructor) {
  ::chai::expt::ArrayPointer<int, TestArrayManager> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(ArrayPointer, ManagerConstructor) {
  ::chai::expt::ArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ArrayPointer, CopyConstructor) {
  ::chai::expt::ArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  ::chai::expt::ArrayPointer<int, TestArrayManager> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ArrayPointer, ConvertingConstructor) {
  ::chai::expt::ArrayPointer<int, TestArrayManager> a(new TestArrayManager());
  ::chai::expt::ArrayPointer<const int, TestArrayManager> b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.data(), nullptr);
  b.free();
}

TEST(ArrayPointer, CopyAssignmentOperator) {
  ::chai::expt::ArrayPointer<int, TestArrayManager> a;
  a = ::chai::expt::ArrayPointer<int, TestArrayManager>(new TestArrayManager());
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
  a.free();
}

TEST(ArrayPointer, resize) {
  const std::size_t n = 10;
  ::chai::expt::ArrayPointer<int, TestArrayManager> a;
  a.resize(n);
  EXPECT_EQ(a.size(), n);
  EXPECT_NE(a.data(), nullptr);
  a.free();
}

TEST(ArrayPointer, data) {
  const std::size_t n = 10;
  ::chai::expt::ArrayPointer<int, TestArrayManager> a;
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
