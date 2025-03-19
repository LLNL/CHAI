//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
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

#include "chai/ManagedArray.hpp"

TEST(ManagedArray, DefaultConstructor)
{
  chai::ManagedArray<float> array;
  ASSERT_EQ(array.size(), 0u);
}

TEST(ManagedArray, SizeConstructor)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

TEST(ManagedArray, SpaceConstructorCPU)
{
  chai::ManagedArray<float> array(10, chai::CPU);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
TEST(ManagedArray, SpaceConstructorGPU)
{
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

#if defined(CHAI_ENABLE_UM)
TEST(ManagedArray, SpaceConstructorUM)
{
  chai::ManagedArray<float> array(10, chai::UM);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}
#endif
#endif
