//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ManagedArrayView.hpp"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdlib>

namespace {
  /**
   * Minimal "ManagerType" for exercising ManagedArrayView in unit tests.
   *
   * Requirements satisfied (as used by ManagedArrayView):
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

      explicit TestArrayManager(std::size_t size)
      {
        m_size = size;
        m_data = static_cast<ElementType*>(std::realloc(m_data, size*sizeof(ElementType)));
      }

      ~TestArrayManager()
      {
        std::free(m_data);
      }

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
}  // anonymous namespace

template <typename ElementType>
using TestArrayView = ::chai::expt::ManagedArrayView<ElementType, TestArrayManager<ElementType>>;

template <typename ElementType>
using ConstTestArrayView = ::chai::expt::ManagedArrayView<const ElementType, TestArrayManager<ElementType>>;

TEST(ManagedArrayView, DefaultConstructor) {
  TestArrayView<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(ManagedArrayView, ManagerConstructor) {
  TestArrayManager<int> manager;
  TestArrayView<int> a{manager};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);
}

TEST(ManagedArrayView, CopyConstructor) {
  TestArrayManager<int> manager{10};
  TestArrayView<int> a{manager};
  TestArrayView<int> b(a);
  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());
}

TEST(ManagedArrayView, ConvertingConstructor) {
  TestArrayManager<int> manager{10};
  TestArrayView<int> a{manager};
  ConstTestArrayView<int> b(a);
  EXPECT_EQ(b.size(), a.size());
  EXPECT_EQ(b.data(), a.data());
}

TEST(ManagedArrayView, CopyAssignmentOperator) {
  TestArrayView<int> a;
  TestArrayManager<int> manager{10};
  a = TestArrayView<int>(manager);
  EXPECT_EQ(a.size(), 10);
  EXPECT_NE(a.data(), nullptr);
}

TEST(ManagedArrayView, Data) {
  const std::size_t n = 10;
  TestArrayManager<int> manager{n};
  TestArrayView<int> a{manager};
  a.update();

  for (std::size_t i = 0; i < n; ++i)
  {
    a[i] = static_cast<int>(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < n; ++i)
  {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
}

TEST(ManagedArrayView, LambdaCapture) {
  const std::size_t n = 10;
  TestArrayManager<int> manager{n};
  TestArrayView<int> a{manager};

  auto f = [=] (std::size_t i)
  {
    a[i] = static_cast<int>(i);
  };

  for (std::size_t i = 0; i < n; ++i)
  {
    f(i);
  }

  int* data = a.data();

  for (std::size_t i = 0; i < n; ++i)
  {
    EXPECT_EQ(data[i], static_cast<int>(i));
  }
}

TEST(ManagedArrayView, ReflectsManagerResizeAfterUpdate) {
  TestArrayManager<int> manager{4};
  TestArrayView<int> a{manager};

  EXPECT_EQ(a.size(), 4);

  manager.resize(7);
  a.update();

  EXPECT_EQ(a.size(), 7);
  EXPECT_EQ(a.data(), manager.data());
}
