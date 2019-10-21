//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X##Y();    \
  TEST(X, Y) { gpu_test_##X##Y(); } \
  static void gpu_test_##X##Y()

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"
#include "chai/config.hpp"

GPU_TEST(ManagedArray, Simple)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::devices::Context host{camp::devices::Host{}};
  camp::devices::Context device{camp::devices::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = i;
  });

  forall(&device, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = array[i] * 2.0;
  });

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
    EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });
}

GPU_TEST(ManagedArray, SimpleWithAsyncMoveFrom)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::devices::Context host{camp::devices::Host{}};
  camp::devices::Context device{camp::devices::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = i;
  });

  forall(&device, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = array[i] * 2.0;
  });

  array.move(chai::CPU, &device);

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });
}

GPU_TEST(ManagedArray, SimpleWithAsyncMoveTo)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::devices::Context host{camp::devices::Host{}};
  camp::devices::Context device{camp::devices::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = i;
  });

  array.move(chai::GPU, &device);

  forall(&device, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = array[i] * 2.0;
  });

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });
}
