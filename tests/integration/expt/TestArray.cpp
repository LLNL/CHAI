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

#ifdef NDEBUG

#ifdef CHAI_ENABLE_CUDA
#define device_assert(EXP) if( !(EXP) ) asm ("trap;")
#else
#define device_assert(EXP) if( !(EXP) ) asm ("s_trap 1;")
#endif

#else
#define device_assert(EXP) assert(EXP)
#endif

#ifdef CHAI_DISABLE_RM
#define assert_empty_map(IGNORED)
#else
#define assert_empty_map(IGNORED) ASSERT_EQ(chai::ArrayManager::getInstance()->getPointerMap().size(),0)
#endif


#include "chai/config.hpp"

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"

#include "umpire/ResourceManager.hpp"


TEST(Array, HostManager)
{
  chai::Array<int> a = chai::makeArray<int, HostManager>(10, allocator);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();

  assert_empty_map(true);
}


GPU_TEST(ManagedArray, PickandSetDeviceToDeviceUM)
{
  chai::ManagedArray<int> array1(10, chai::UM);
  chai::ManagedArray<int> array2(10, chai::UM);

  forall(gpu(), 0, 10, [=] __device__(int i) { array1[i] = i; });

  forall(gpu(), 0, 10, [=] __device__(int i) {
    int temp = array1.pick(i);
    array2.set(i, temp);
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array2[i], i); });

  array1.free();
  array2.free();
  assert_empty_map(true);
}

