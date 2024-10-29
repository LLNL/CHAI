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

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"

#include "umpire/ResourceManager.hpp"

GPU_TEST(PinnedArray, PodTestGPU)
{
  const int n = 10000;
  chai::experimental::PinnedArray<double> a(n);

  forall(gpu(), 0, n, [=] CHAI_DEVICE (int i) {
    a[i] = (double) i;
  });

  forall(sequential(), 0, n, [=] CHAI_HOST (int i) {
    ASSERT_EQ(a[i], i);
  });

  array.free();
}
