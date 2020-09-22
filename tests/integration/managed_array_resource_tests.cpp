//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
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
#include "../src/util/gpu_clock.hpp"

#include "chai/ManagedArray.hpp"
#include "chai/config.hpp"

#ifdef CHAI_ENABLE_CUDA
GPU_TEST(ManagedArray, Simple)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::resources::Resource host{camp::resources::Host{}};
  camp::resources::Resource device{camp::resources::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = i;
  });

  forall(&device, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = array[i] * 2.0;
  });

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
    EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });

  array.free();
}

GPU_TEST(ManagedArray, SimpleWithAsyncMoveFrom)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::resources::Resource host{camp::resources::Host{}};
  camp::resources::Resource device{camp::resources::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = i;
  });

  forall(&device, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = array[i] * 2.0;
  });

  array.move(chai::CPU, &device);

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });
}

GPU_TEST(ManagedArray, SimpleWithAsyncMoveTo)
{
  constexpr std::size_t ARRAY_SIZE{1024};

  camp::resources::Resource host{camp::resources::Host{}};
  camp::resources::Resource device{camp::resources::Cuda{}};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = i;
  });

  array.move(chai::GPU, &device);

  forall(&device, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array[i] = array[i] * 2.0;
  });

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      EXPECT_DOUBLE_EQ(array[i], i*2.0);
  });

  array.free();
}

GPU_TEST(ManagedArray, MultiStreamDepends)
{
  constexpr std::size_t ARRAY_SIZE{1024};
  int clockrate{get_clockrate()}; 

  camp::resources::Resource dev1{camp::resources::Cuda{}};
  camp::resources::Resource dev2{camp::resources::Cuda{}};
  camp::resources::Resource host{camp::resources::Host{}};

  chai::ManagedArray<double> array1(ARRAY_SIZE);
  chai::ManagedArray<double> array2(ARRAY_SIZE);

  forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array1[i] = i;
      gpu_time_wait_for(10, clockrate);
  });

  auto e2 = forall(&dev2, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array2[i] = -1;
      gpu_time_wait_for(20, clockrate);
  });

  dev1.wait_for(&e2);

  forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array1[i] *= array2[i];
      gpu_time_wait_for(10, clockrate);
  });

  array1.move(chai::CPU, &dev1);

  forall(&host, 0, 10, [=] CHAI_HOST_DEVICE (int i) {
      EXPECT_DOUBLE_EQ(array1[i], i*-1.0);
  });

  array1.free();
  array2.free();
}

GPU_TEST(ManagedArray, MultiStreamSingleArray)
{
  constexpr std::size_t ARRAY_SIZE{1024};
  int clockrate{get_clockrate()}; 

  chai::ManagedArray<double> array1(ARRAY_SIZE);

  camp::resources::Resource dev1{camp::resources::Cuda{}};
  camp::resources::Resource dev2{camp::resources::Cuda{}};


  auto e1 = forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      if (i % 2 == 0) {
        array1[i] = i;
        gpu_time_wait_for(10, clockrate);
      }
  });

  auto e2 = forall(&dev2, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      if (i % 2 == 1) {
        gpu_time_wait_for(20, clockrate);
        array1[i] = i;
      }
  });

  array1.move(chai::CPU, &dev1);

  camp::resources::Resource host{camp::resources::Host{}};

  forall(&host, 0, 10, [=] CHAI_HOST_DEVICE (int i) {
      EXPECT_DOUBLE_EQ(array1[i], (double)i);
  });
}
#endif //#ifdef CHAI_ENABLE_CUDA
