//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
///
/// Source file containing tests for CHAI with basic RAJA constructs
///
#include "RAJA/RAJA.hpp"

#include "chai/ManagedArray.hpp"
#include "chai/ManagedArrayView.hpp"

#include <iostream>

#include "gtest/gtest.h"

// TODO: add hip policy for these tests.
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

  float* raw_v2 = v2.data();
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(raw_v2[i], i * 2.0f * 2.0f);
    ;
  }
  
  v1.free();
  v2.free();
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

  float* raw_v2 = v2_array.data();
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(raw_v2[i], i * 1.0f * 2.0f * 2.0f);
    ;
  }

  v1_array.free();
  v2_array.free();
}

#if defined(CHAI_ENABLE_MULTIVIEW_TEST)
CUDA_TEST(ChaiTest, MultiView)
{
  chai::ManagedArray<float> v1_array(10);
  chai::ManagedArray<float> v2_array(10);

  // TODO: This stack array gets converted to a raw host pointer, which appears
  //       to be dereferenced on the device when accessing the multiview.
  chai::ManagedArray<float> all_arrays[2];
  all_arrays[0] = v1_array;
  all_arrays[1] = v2_array;

  // default MultiView
  using view = chai::ManagedArrayMultiView<float, RAJA::Layout<1>>;
  view mview(all_arrays, RAJA::Layout<1>(10));

  // MultiView with index in 1st position
  using view1p = chai::ManagedArrayMultiView<float, RAJA::Layout<1>, 1>;
  view1p mview1p(all_arrays, RAJA::Layout<1>(10));

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    mview(0,i) = static_cast<float>(i * 1.0f);
  });

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    // use both MultiViews
    mview(1,i) = mview1p(i,0) * 2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 10), [=](int i) {
    ASSERT_FLOAT_EQ(mview(1,i), i * 2.0f);
  });

  RAJA::forall<parallel_raja_policy>(RAJA::RangeSegment(0, 10), [=] PARALLEL_RAJA_DEVICE(int i) {
    mview(1,i) *= 2.0f;
  });

  // accessing pointer to v2_array
  float* raw_v2 = mview.data[1].data();
  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(raw_v2[i], i * 1.0f * 2.0f * 2.0f);
    ;
  }

  v1_array.free();
  v2_array.free();
}
#endif  // defined(CHAI_ENABLE_MULTIVIEW_TEST)
