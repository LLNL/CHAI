//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X##Y();    \
  TEST(X, Y) { gpu_test_##X##Y(); } \
  static void gpu_test_##X##Y()

#include "chai/config.hpp"

#include "chai/containers/PinnedArray.hpp"

TEST(PinnedArray, DefaultConstructor)
{
  chai::experimental::PinnedArray<float> array;
  ASSERT_EQ(array.size(), 0u);
}

TEST(PinnedArray, SizeConstructor)
{
  chai::PinnedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}
