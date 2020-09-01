//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for CHAI with basic RAJA constructs
///
#include "RAJA/RAJA.hpp"

#include "chai/ManagedArray.hpp"
#include "chai/ManagedArrayView.hpp"

#include <iostream>

#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)                 \
  static void cuda_test_##X##_##Y();    \
  TEST(X, Y) { cuda_test_##X##_##Y(); } \
  static void cuda_test_##X##_##Y()

#if defined(RAJA_ENABLE_CUDA)
#define PARALLEL_RAJA_DEVICE __device__
  using parallel_raja_policy = RAJA::cuda_exec<16>;
#elif defined(RAJA_ENABLE_OPENMP)
#define PARALLEL_RAJA_DEVICE
  using parallel_raja_policy = RAJA::omp_parallel_for_exec;
#else
#define PARALLEL_RAJA_DEVICE
  using parallel_raja_policy = RAJA::seq_exec;
#endif

CUDA_TEST(ChaiTest, Simple)
{
  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    v1[i] = static_cast<float>(i * 1.0f);
  });

  std::cout << "end of loop 1" << std::endl;

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    v2[i] = v1[i] * 2.0f;
  });

  std::cout << "end of loop 2" << std::endl;

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    ASSERT_FLOAT_EQ(v2[i], i * 2.0f);
  });

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    v2[i] *= 2.0f;
  });

  float* raw_v2 = v2;
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(raw_v2[i], i * 2.0f * 2.0f);
    ;
  }
}

CUDA_TEST(ChaiTest, Views)
{
  chai::ManagedArray<float> v1_array(10);
  chai::ManagedArray<float> v2_array(10);

  using view = chai::ManagedArrayView<float, RAJA::Layout<1> >;

  view v1(v1_array, 10);
  view v2(v2_array, 10);

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    v1(i) = static_cast<float>(i * 1.0f);
  });

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    v2(i) = v1(i) * 2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    ASSERT_FLOAT_EQ(v2(i), i * 2.0f);
  });

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    v2(i) *= 2.0f;
  });

  float* raw_v2 = v2.data;
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(raw_v2[i], i * 1.0f * 2.0f * 2.0f);
    ;
  }
}
