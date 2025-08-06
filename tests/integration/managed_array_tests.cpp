//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/config.hpp"

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"

#include "umpire/ResourceManager.hpp"

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

struct my_point {
  double x;
  double y;
};

TEST(ManagedArray, SetOnHost)
{
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();

  assert_empty_map(true);
}

#if (!defined(CHAI_DISABLE_RM))
TEST(ManagedArray, Const)
{
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const float> array_const(array);

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array_const[i], i); });

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, UserCallbackHost)
{
   bool callbackCalled = false;

   chai::ManagedArray<float> array(10);
   array.setUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                           callbackCalled = true;
                         });

   array.free();
   ASSERT_TRUE(callbackCalled);
}

#endif

TEST(ManagedArray, Slice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  const int SLICE_SZ = 5;
  chai::ManagedArray<float> sl = array.slice(0,SLICE_SZ);
  ASSERT_EQ(SLICE_SZ, sl.size());

  sl.free();
  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, SliceCopyCtor) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  const int SLICE_SZ = 5;
  chai::ManagedArray<float> sl = array.slice(0,SLICE_SZ);
  chai::ManagedArray<float> slcopy = sl;
  ASSERT_EQ(SLICE_SZ, slcopy.size());

  sl.free();
  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, SliceOfSlice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  const int SLICE_SZ_1 = 6;
  const int SLICE_SZ_2 = 3;
  chai::ManagedArray<float> sl1 = array.slice(0,6);
  chai::ManagedArray<float> sl2 = sl1.slice(3,3);
  ASSERT_EQ(sl1.size(), SLICE_SZ_1);
  ASSERT_EQ(sl2.size(), SLICE_SZ_2);

  forall(sequential(), 0, 3, [=] (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 3, [=] (int i) {
    ASSERT_EQ(array[i], array[i+3]);
  });

  sl1.free();
  sl2.free();
  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, ArrayOfSlices) {
  chai::ManagedArray<float> array(10);
  chai::ManagedArray<chai::ManagedArray<float>> arrayOfSlices(5);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 5, [=] (int i) {
      arrayOfSlices[i] = array.slice(2*i, 2);
      arrayOfSlices[i][1] = arrayOfSlices[i][0];
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(arrayOfSlices[i].size(), 2);
    ASSERT_EQ(array[2*i], array[2*i+1]);
  });

  arrayOfSlices.free();
  array.free();
  assert_empty_map(true);
}

#if (!defined(CHAI_DISABLE_RM))
TEST(ManagedArray, PickHostFromHostConst) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}
#endif

TEST(ManagedArray, PickHostFromHost)
{
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, SetHostToHost)
{
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  ASSERT_EQ(array[5], 10);

  array.free();
  assert_empty_map(true);
}


#if defined(CHAI_ENABLE_UM)
#if (!defined(CHAI_DISABLE_RM))
TEST(ManagedArray, PickHostFromHostConstUM) {
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}
#endif

TEST(ManagedArray, PickHostFromHostUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, SetHostToHostUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  ASSERT_EQ(array[5], 10);

  array.free();
  assert_empty_map(true);
}

#endif

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

#if defined(CHAI_ENABLE_UM)
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

GPU_TEST(ManagedArray, PickHostFromDeviceUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}

#if (!defined(CHAI_DISABLE_RM))
GPU_TEST(ManagedArray, PickHostFromDeviceConstUM) {
  chai::ManagedArray<int> array(10, chai::UM);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}
#endif

GPU_TEST(ManagedArray, SetHostToDeviceUM)
{
  chai::ManagedArray<int> array(10);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  temp = array.pick(5);
  ASSERT_EQ(temp, 10);

  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, PickandSetSliceDeviceToDeviceUM) {
  chai::ManagedArray<int> array(10, chai::UM);
  chai::ManagedArray<int> sl1 = array.slice(0,5);
  chai::ManagedArray<int> sl2 = array.slice(5,5);

  forall(gpu(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(gpu(), 0, 5, [=] __device__ (int i) {
      int temp = sl2.pick(i);
      temp += sl2.pick(i);
      sl1.set(i, temp);
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], (i+5)*2);
  });

  array.free();
  assert_empty_map(true);
}
#endif

#if (!defined(CHAI_DISABLE_RM))
GPU_TEST(ManagedArray, PickandSetDeviceToDevice)
{
  chai::ManagedArray<int> array1(10);
  chai::ManagedArray<int> array2(10);

  forall(gpu(), 0, 10, [=] __device__(int i) { array1[i] = i; });

  chai::ManagedArray<const int> array_const(array1);

  forall(gpu(), 0, 10, [=] __device__(int i) {
    int temp = array1.pick(i);
    temp += array_const.pick(i);
    array2.set(i, temp);
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array2[i], i + i); });

  array1.free();
  array2.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, PickandSetSliceDeviceToDevice) {
  chai::ManagedArray<int> array(10);
  chai::ManagedArray<int> sl1 = array.slice(0,5);
  chai::ManagedArray<int> sl2 = array.slice(5,5);

  forall(gpu(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(gpu(), 0, 5, [=] __device__ (int i) {
      int temp = sl2.pick(i);
      temp += sl2.pick(i);
      sl1.set(i, temp);
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], (i+5)*2);
  });

  array.free();
  assert_empty_map(true);
}


GPU_TEST(ManagedArray, PickHostFromDevice)
{
  chai::ManagedArray<int> array(10);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, PickHostFromDeviceConst)
{
  chai::ManagedArray<int> array(10);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array_const[i], i); });

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, SetHostToDevice)
{
  chai::ManagedArray<int> array(10);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  temp = array.pick(5);
  ASSERT_EQ(temp, 10);

  array.free();
  assert_empty_map(true);
}

#endif

GPU_TEST(ManagedArray, ArrayOfSlicesDevice) {
  chai::ManagedArray<float> array(10);
  chai::ManagedArray<chai::ManagedArray<float>> arrayOfSlices(5);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 5, [=] (int i) {
      arrayOfSlices[i] = array.slice(2*i, 2);
  });

  forall(gpu(), 0, 5, [=] __device__ (int i) {
      arrayOfSlices[i][1] = arrayOfSlices[i][0];
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(arrayOfSlices[i].size(), 2);
    ASSERT_EQ(array[2*i], array[2*i+1]);
  });

  arrayOfSlices.free();
  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, SliceOfSliceDevice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl1 = array.slice(0,6);
  chai::ManagedArray<float> sl2 = sl1.slice(3,3);

  forall(gpu(), 0, 3, [=] __device__ (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 3, [=] (int i) {
    ASSERT_EQ(array[i], array[i+3]);
  });

  sl1.free();
  sl2.free();
  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, SliceDevice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl1 = array.slice(0,5);
  chai::ManagedArray<float> sl2 = array.slice(5,5);

  forall(gpu(), 0, 5, [=] __device__ (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], array[i+5]);
  });

  sl1.free();
  sl2.free();
  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, SetOnDevice) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] *= 2; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], 2 * i); });

  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, GetGpuOnHost)
{
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_UM)
GPU_TEST(ManagedArray, SetOnDeviceUM)
{
  chai::ManagedArray<float> array(10, chai::UM);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
  assert_empty_map(true);
}
#endif
#endif

TEST(ManagedArray, Allocate)
{
  chai::ManagedArray<float> array;
  ASSERT_EQ(array.size(), 0u);

  array.allocate(10);
  ASSERT_EQ(array.size(), 10u);

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, ReallocateCPU)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  array.reallocate(20);
  ASSERT_EQ(array.size(), 20u);

  forall(sequential(), 0, 20, [=](int i) {
    if (i < 10) {
      ASSERT_EQ(array[i], i);
    } else {
      array[i] = i;
      ASSERT_EQ(array[i], i);
    }
  });

  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

GPU_TEST(ManagedArray, ReallocateGPU)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  array.reallocate(20);
  ASSERT_EQ(array.size(), 20u);

  forall(gpu(), 0, 20, [=]__device__(int i) {
    if (i < 10) {
      device_assert(array[i] == i);
    } else {
      array[i] = i;
      device_assert(array[i] == i);
    }
  });

  array.free();
  assert_empty_map(true);
}

#endif

TEST(ManagedArray, NullpointerConversions)
{
  chai::ManagedArray<float> a;
  a.free();
  a = nullptr;

  chai::ManagedArray<const float> b;
  b.free();
  b = nullptr;

  ASSERT_EQ(a.size(), 0u);
  ASSERT_EQ(b.size(), 0u);

  chai::ManagedArray<float> c(nullptr);

  ASSERT_EQ(c.size(), 0u);
  assert_empty_map(true);
}

TEST(ManagedArray, PodTest)
{
  chai::ManagedArray<my_point> array(1);

  forall(sequential(), 0, 1, [=](int i) {
    array[i].x = (double)i;
    array[i].y = (double)i * 2.0;
  });

  forall(sequential(), 0, 1, [=](int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i * 2.0);
  });

  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
GPU_TEST(ManagedArray, PodTestGPU)
{
  chai::ManagedArray<my_point> array(1);

  forall(gpu(), 0, 1, [=] __device__(int i) {
    array[i].x = (double)i;
    array[i].y = (double)i * 2.0;
  });

  forall(sequential(), 0, 1, [=](int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i * 2.0);
  });

  array.free();
  assert_empty_map(true);
}
#endif

TEST(ManagedArray, ExternalConstructorUnowned)
{
  float* data = static_cast<float*>(std::malloc(100 * sizeof(float)));

  for (int i = 0; i < 100; i++) {
    data[i] = 1.0f * i;
  }

  chai::ManagedArray<float> array =
      chai::makeManagedArray<float>(data, 100, chai::CPU, false);

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(data[i], array[i]); });

  array.free();

  for (int i = 0; i < 100; i++) {
    ASSERT_EQ(data[i], 1.0f * i);
  }

  std::free(data);
  assert_empty_map(true);
}

TEST(ManagedArray, ExternalConstructorOwned)
{
  float* data = static_cast<float*>(std::malloc(20 * sizeof(float)));

  for (int i = 0; i < 20; i++) {
    data[i] = 1.0f * i;
  }

  chai::ManagedArray<float> array =
      chai::makeManagedArray<float>(data, 20, chai::CPU, true);

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(data[i], array[i]); });

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, ExternalOwnedFromManagedArray)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 1.0f * i; });

  chai::ManagedArray<float> arrayCopy =
      chai::makeManagedArray<float>(array.data(chai::CPU), 20, chai::CPU, true);

  ASSERT_EQ(array.data(), arrayCopy.data());

  // should be able to free through the new ManagedArray
  arrayCopy.free();
  assert_empty_map(true);
}

TEST(ManagedArray, ExternalUnownedFromManagedArray)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 1.0f * i; });

  chai::ManagedArray<float> arrayCopy =
      chai::makeManagedArray<float>(array.data(chai::CPU), 20, chai::CPU, false);

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(arrayCopy[i], 1.0f * i); });
  // freeing from an unowned pointer should leave the original ManagedArray intact
  arrayCopy.free();
  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, ExternalUnownedMoveToGPU)
{
  float data[20];
  for (int i = 0; i < 20; i++) {
    data[i] = 0.;
  }

  chai::ManagedArray<float> array =
      chai::makeManagedArray<float>(data, 20, chai::CPU, false);

  forall(gpu(), 0, 20, [=] __device__ (int i) { array[i] = 1.0f * i; });

  forall(sequential(), 0, 20, [=] (int i) { ASSERT_EQ(array[i], 1.0f * i); });

  array.free();
  assert_empty_map(true);
}
#endif
#endif

TEST(ManagedArray, data)
{
  int length = 10;
  chai::ManagedArray<int> array(length);

  forall(sequential(), 0, length, [=] (int i) {
    array[i] = i;
  });

  int* data = array.data();

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data[i], i);
    data[i] = length - 1 - i;
  }

  forall(sequential(), 0, length, [=] (int i) {
    EXPECT_EQ(array[i], length - 1 - i);
  });

  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, dataGPU)
{
  // Initialize
  int transfersH2D = 0;
  int transfersD2H = 0;

  int length = 10;
  chai::ManagedArray<int> array;
  array.allocate(length,
                 chai::GPU,
                 [&] (const chai::PointerRecord*, chai::Action act, chai::ExecutionSpace s) {
                   if (act == chai::ACTION_MOVE) {
                     if (s == chai::CPU) {
                       ++transfersD2H;
                     }
                     else if (s == chai::GPU) {
                       ++transfersH2D;
                     }
                   }
                 });

  forall(gpu(), 0, length, [=] __device__ (int i) {
    int* d_data = array.data();
    d_data[i] = i;
  });

  // Move data to host with touch
  int* data = array.data();

  EXPECT_EQ(transfersD2H, 1);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data[i], i);
    data[i] = length - 1 - i;
  }

  // Move data to device with touch
  forall(gpu(), 0, length, [=] __device__ (int i) {
    array.data();
    array[i] += 1;
  });

  EXPECT_EQ(transfersH2D, 1);

  // Move data to host without touch
  chai::ManagedArray<const int> array2 = array;
  const int* data2 = array2.data();

  EXPECT_EQ(transfersD2H, 2);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data2[i], length - i);
  }

  // Access on device with touch (should not be moved)
  forall(gpu(), 0, length, [=] __device__ (int i) {
    array.data();
    array[i] += i;
  });

  EXPECT_EQ(transfersH2D, 1);

  // Move data to host
  forall(sequential(), 0, length, [=] (int i) {
    EXPECT_EQ(array[i], length);
  });

  EXPECT_EQ(transfersD2H, 3);
  EXPECT_EQ(transfersH2D, 1);

  array.free();
  assert_empty_map(true);
}
#endif
#endif

TEST(ManagedArray, cdata)
{
  int length = 10;
  chai::ManagedArray<int> array(length);

  forall(sequential(), 0, length, [=] (int i) {
    array[i] = i;
  });

  const int* data = array.cdata();

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data[i], i);
  }

  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, cdataGPU)
{
  // Initialize
  int transfersH2D = 0;
  int transfersD2H = 0;

  int length = 10;
  chai::ManagedArray<int> array;
  array.allocate(length,
                 chai::GPU,
                 [&] (const chai::PointerRecord*, chai::Action act, chai::ExecutionSpace s) {
                   if (act == chai::ACTION_MOVE) {
                     if (s == chai::CPU) {
                       ++transfersD2H;
                     }
                     else if (s == chai::GPU) {
                       ++transfersH2D;
                     }
                   }
                 });

  forall(gpu(), 0, length, [=] __device__ (int i) {
    const int* d_data = array.cdata();

    if (d_data[i] == array[i]) {
      array[i] = i;
    }
  });

  // Move data to host without touch
  const int* data = array.cdata();

  EXPECT_EQ(transfersD2H, 1);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data[i], i);
  }

  // Access on device with touch (should not be moved)
  forall(gpu(), 0, length, [=] __device__ (int i) {
    const int* d_data = array.cdata();

    if (d_data[i] == array[i]) {
       array[i] += 1;
    }
  });

  EXPECT_EQ(transfersH2D, 0);

  // Move data to host without touch
  chai::ManagedArray<const int> array2 = array;
  const int* data2 = array2.cdata();

  EXPECT_EQ(transfersD2H, 2);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(data2[i], i + 1);
  }

  // Access on device with touch (should not be moved)
  forall(gpu(), 0, length, [=] __device__ (int i) {
    const int* d_data = array.cdata();

    if (d_data[i] == array[i]) {
       array[i] += 1;
    }
  });

  EXPECT_EQ(transfersH2D, 0);

  // Move data to host with touch
  forall(sequential(), 0, length, [=] (int i) {
    EXPECT_EQ(array[i], i + 2);
  });

  EXPECT_EQ(transfersD2H, 3);
  EXPECT_EQ(transfersH2D, 0);

  array.free();
  assert_empty_map(true);
}
#endif
#endif

TEST(ManagedArray, Iterators)
{
  int length = 10;
  chai::ManagedArray<int> array(length);

  forall(sequential(), 0, length, [=] (int i) {
    array[i] = i;
  });

  // Make sure the iterator distance is the size of the array
  EXPECT_EQ(std::distance(array.begin(), array.end()), length);

  // Double each element with a range-based for loop
  for (int& val : array)
  {
    val *= 2;
  }

  // Double each element again with an <algorithm>
  std::for_each(array.begin(), array.end(), [](int& val) { val *= 2; });

  // Make sure a reference to a const array can be iterated over
  const chai::ManagedArray<int>& const_array = array;
  int i = 0;
  for (const int val : const_array)
  {
    EXPECT_EQ(val, i * 4);
    i++;
  }

  array.free();
  assert_empty_map(true);
}

TEST(ManagedArray, Reset)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 1.0f * i; });

  array.reset();
  array.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, ResetDevice)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 0.0f; });

  forall(gpu(), 0, 20, [=] __device__(int i) { array[i] = 1.0f * i; });

  array.reset();

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(array[i], 0.0f); });

  array.free();
  assert_empty_map(true);
}
#endif
#endif


#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, UserCallback)
{
  int num_h2d = 0;
  int num_d2h = 0;
  size_t bytes_h2d = 0;
  size_t bytes_d2h = 0;
  size_t bytes_alloc = 0;
  size_t bytes_free = 0;

  chai::ManagedArray<float> array;
  array.allocate(20,
                 chai::CPU,
                 [&] (const chai::PointerRecord* record, chai::Action act, chai::ExecutionSpace s) {
                    const size_t bytes = record->m_size;
                    printf("cback: act=%d, space=%d, bytes=%ld\n",
                      (int)act, (int)s, (long)bytes);
                   if (act == chai::ACTION_MOVE) {
                     if (s == chai::CPU) {
                       ++num_d2h;
                       bytes_d2h += bytes;
                     }
                     if (s == chai::GPU) {
                       ++num_h2d;
                       bytes_h2d += bytes;
                     }
                   }
                   if (act == chai::ACTION_ALLOC) {
                     bytes_alloc += bytes;
                   }
                   if (act == chai::ACTION_FREE) {
                     bytes_free += bytes;
                   }
                 });

  for (int iter = 0; iter < 10; ++iter) {
    forall(sequential(), 0, 20, [=](int i) { array[i] = 0.0f; });

    forall(gpu(), 0, 20, [=] __device__(int i) { array[i] = 1.0f * i; });
  }


  ASSERT_EQ(num_d2h, 9);
  ASSERT_EQ(bytes_d2h, 9 * sizeof(float) * 20);
  ASSERT_EQ(num_h2d, 10);
  ASSERT_EQ(bytes_h2d, 10 * sizeof(float) * 20);

  array.free();

  ASSERT_EQ(bytes_alloc, 2 * 20 * sizeof(float));
  ASSERT_EQ(bytes_free, 2 * 20 * sizeof(float));
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, CallBackConst)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&](const chai::PointerRecord* record, chai::Action act, chai::ExecutionSpace s)
  {
    const size_t bytes = record->m_size;
    printf("cback: act=%d, space=%d, bytes=%ld\n", (int) act, (int) s, (long) bytes);
    if (act == chai::ACTION_MOVE)
    {
      if (s == chai::CPU)
      {
        printf("Moved to host\n");
        ++num_d2h;
      }
      else if (s == chai::GPU)
      {
        printf("Moved to device\n");
        ++num_h2d;
      }
    }
    if (act == chai::ACTION_FOUND_ABANDONED) {
       printf("in abandoned!\n");
       ASSERT_EQ(false,true);
    }
  };

  chai::ManagedArray<int> array(100);
  array.setUserCallback(callBack);

  // Set the values.
  forall(sequential(), 0, 100, [=](int i) { array[i] = i; });

  // Create a const copy and move it to the device.
  chai::ManagedArray<int const> array_const = array;

  /* array to store error flags on device */
  chai::ManagedArray<int> error_flag(100);

  forall(gpu(), 0, 100, [=] __device__(int i) { error_flag[i] = (array_const[i] == i) ? 0 : 1; });

  // Check error flags on host
  forall(sequential(), 0, 100, [=](int i) { ASSERT_EQ(error_flag[i], 0); });
  error_flag.free();

  // Change the values, use a for-loop so as to not trigger a movement.
  for (int i = 0; i < 100; ++i)
  {
    array[i] = 2 * i;
  }

  // Capture the array on host, should not trigger a movement.
  forall(sequential(), 0, 100, [=](int i) { ASSERT_EQ(array[i], 2 * i); });

  ASSERT_EQ(num_h2d, 1);
  ASSERT_EQ(num_d2h, 0);

  array.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, CallBackConstArray)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&] (const chai::PointerRecord* record, chai::Action act, chai::ExecutionSpace s)
  {
    const size_t bytes = record->m_size;
    printf("cback: act=%d, space=%d, bytes=%ld\n", (int) act, (int) s, (long) bytes);
    if (act == chai::ACTION_MOVE)
    {
      if (s == chai::CPU)
      {
        printf("Moved to host\n");
        ++num_d2h;
      }
      else if (s == chai::GPU)
      {
        printf("Moved to device\n");
        ++num_h2d;
      }
    }
    if (act == chai::ACTION_FOUND_ABANDONED) {
       printf("in abandoned!\n");
       ASSERT_TRUE(false);
    }
  };

  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);
  outerArray.setUserCallback(callBack);

  /* array to store error flags on device */
  chai::ManagedArray<chai::ManagedArray<int>> outerErrorArray(N); 

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);
      temp.setUserCallback(callBack);

      chai::ManagedArray<int> errorTemp(N);

      forall(sequential(), 0, N, [=](int j) {
        temp[j] = N * i + j;
      });

      outerArray[i] = temp;
      outerErrorArray[i] = errorTemp;
    }
  );

  // Create a const copy and move it to the device.
  chai::ManagedArray<chai::ManagedArray<int> const> outerArrayConst = outerArray;
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        outerErrorArray[i][j] = (outerArrayConst[i][j] == N * i + j) ? 0 : 1;
      }
    }
  );

  // Check error flags on host
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerErrorArray[i][j], 0);
      }
    }
  );

  // Capture the array on host, should not trigger a movement of the outer array.
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerArray[i][j], N * i + j);
      }
    }
  );

  ASSERT_EQ(num_h2d, N + 1);
  ASSERT_EQ(num_d2h, N);

  for (int i = 0; i < N; ++i) {
    outerArray[i].free();
    outerErrorArray[i].free();
  }

  outerArray.free();
  outerErrorArray.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, CallBackConstArrayConst)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&] (const chai::PointerRecord* record, chai::Action act, chai::ExecutionSpace s)
  {
    const size_t bytes = record->m_size;
    printf("cback: act=%d, space=%d, bytes=%ld\n", (int) act, (int) s, (long) bytes);
    if (act == chai::ACTION_MOVE)
    {
      if (s == chai::CPU)
      {
        printf("Moved to host\n");
        ++num_d2h;
      }
      else if (s == chai::GPU)
      {
        printf("Moved to device\n");
        ++num_h2d;
      }
    }
  };

  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);
  outerArray.setUserCallback(callBack);

  /* array to store error flags on device */
  chai::ManagedArray<chai::ManagedArray<int>> outerErrorArray(N); 

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);
      temp.setUserCallback(callBack);

      chai::ManagedArray<int> errorTemp(N);

      forall(sequential(), 0, N,
        [=](int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
      outerErrorArray[i] = errorTemp;
    }
  );

  // Create a const copy of int const and move it to the device.
  chai::ManagedArray<chai::ManagedArray<int const> const> outerArrayConst = 
    reinterpret_cast<chai::ManagedArray<chai::ManagedArray<int const> const> &>(outerArray);
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        outerErrorArray[i][j] = (outerArrayConst[i][j] == N * i + j) ? 0 : 1;
      }
    }
  );

  // Check error flags on host
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerErrorArray[i][j], 0);
      }
    }
  );

  // Capture the array on host, should not trigger a movement of the outer array.
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerArray[i][j], N * i + j);
      }
    }
  );

  ASSERT_EQ(num_h2d, N + 1);
  ASSERT_EQ(num_d2h, 0);

  for (int i = 0; i < N; ++i) {
    outerArray[i].free();
    outerErrorArray[i].free();
  }

  outerArray.free();
  outerErrorArray.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, DeviceInitializedNestedArrays)
{
  int N = 5;
  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);

  forall(gpu(), 0, N,
    [=]__device__(int i)
    {
      outerArray[i] = nullptr;
    }
  );

  forall(sequential(), 0, N,
    [=](int i)
    {
       outerArray[i] = chai::ManagedArray<int>(1);
    }
  );

  forall(gpu(), 0, N,
    [=]__device__(int i)
    {
      for (int j = 0; j < 1; ++j) {
         outerArray[i][j] = 0;
      }
    }
  );

  outerArray.reallocate(2*N);

  forall(sequential(), N,2*N,
    [=](int i)
    {
       outerArray[i] = chai::ManagedArray<int>(1);
    }
  );

  forall(gpu(), N, 2*N,
    [=]__device__(int i)
    {
      for (int j = 0; j < 1; ++j) {
         outerArray[i][j] = 0;
      }
    }
  );

  forall(sequential(), 0, 2*N,
    [=](int i)
    {
      for (int j = 0; j < 1; ++j) {
        ASSERT_EQ(outerArray[i][j],0);
      }
      outerArray[i].free();
    }
  );
  outerArray.free();
  assert_empty_map(true);
}
#endif
#endif


#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, Move)
{
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  array.move(chai::CPU);

  ASSERT_EQ(array[5], 5);

  array.free();
  assert_empty_map(true);
}

/**
 * This test creates an array of arrays where the outer array is on the
 * CPU and the inner arrays are on the GPU. It then captures the outer array
 * on the CPU and checks that the inner arrays were moved to the CPU.
 */
GPU_TEST(ManagedArray, MoveInnerToHost)
{
  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);

  /* Loop over the outer array and populate it with arrays on the GPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N, chai::GPU);

      forall(gpu(), 0, N,
        [=] __device__(int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
    }
  );

  /* Capture the outer array and check that the values of the inner array are
   * correct. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerArray[i][j], N * i + j);
      }
    }
  );

  for (int i = 0; i < N; ++i) {
    outerArray[i].free();
  }

  outerArray.free();
  assert_empty_map(true);
}

/**
 * This test creates an array of arrays where both the outer array and inner
 * arrays are on the CPU. It then captures the outer array on the GPU and
 * modifies the values of the inner arrays. Finally it captures the outer array
 * on the CPU and checks that the values of the inner arrays were modified
 * correctly. 
 */
GPU_TEST(ManagedArray, MoveInnerToDevice)
{
  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);

      forall(sequential(), 0, N,
        [=](int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
    }
  );

  /* Capture the outer array on the GPU and update the values of the inner
   * arrays. */
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        outerArray[i][j] *= 2;
      }
    }
  );

  /* Capture the outer array on the CPU and check the values of the inner
   * arrays. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerArray[i][j], 2 * (N * i + j));
      }
    }
  );

  for (int i = 0; i < N; ++i) {
    outerArray[i].free();
  }

  outerArray.free();
  assert_empty_map(true);
}

/**
 * This test creates an array of arrays of arrays where all of the arrays begin
 * on the CPU. It then captures the outermost array on the GPU and modifies the
 * values of the innermost arrays. Finally it captures the outermost array
 * on the CPU and checks that the values of the innermost arrays were modified
 * correctly. 
 */
GPU_TEST(ManagedArray, MoveInnerToDevice2)
{
  const int N = 5;

  /* Create the outermost array on the CPU. */
  chai::ManagedArray<chai::ManagedArray<chai::ManagedArray<int>>> outerArray(N);

  forall(sequential(), 0, N,
    [=](int i)
    {
      /* Create the middle array on the CPU. */
      chai::ManagedArray<chai::ManagedArray<int>> middle(N);
      middle.registerTouch(chai::CPU);

      for( int j = 0; j < N; ++j )
      {
        /* Create the innermost array on the CPU. */
        chai::ManagedArray<int> inner(N);
        inner.registerTouch(chai::CPU);

        for( int k = 0; k < N; ++k )
        {
          inner[k] =  N * N * i + N * j + k;
        }

        middle[j] = inner;
      }

      outerArray[i] = middle;
    }
  );

  /* Capture the outermost array on the GPU and update the values of the
   * innermost arrays. */
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        for (int k = 0; k < N; ++k)
        { 
          outerArray[i][j][k] *= 2;
        }
      }
    }
  );

  /* Capture the outermost array on the CPU and check the values of the
   * innermost arrays. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      for( int j = 0; j < N; ++j)
      {
        for (int k = 0; k < N; ++k)
        { 
          ASSERT_EQ(outerArray[i][j][k], 2 * (N * N * i + N * j + k));
        }
      }
    }
  );

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      outerArray[i][j].free();
    }
    outerArray[i].free();
  }
  outerArray.free();
  assert_empty_map(true);
}

GPU_TEST(ManagedArray, MoveInnerToDeviceAgain)
{
  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);

      forall(sequential(), 0, N,
        [=](int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
    }
  );

  /* Capture the outer array on the GPU and update the values of the inner
   * arrays. */
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        outerArray[i][j] *= 2;
      }
    }
  );

  /* Capture the outer array on the GPU and update the values of the inner
   * arrays. This time, the array should already be resident on the GPU. */
  forall(gpu(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        outerArray[i][j] *= 2;
      }
    }
  );

  /* Capture the outer array on the CPU and check the values of the inner
   * arrays. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      for (int j = 0; j < N; ++j)
      {
        ASSERT_EQ(outerArray[i][j], 4 * (N * i + j));
      }
    }
  );

  for (int i = 0; i < N; ++i) {
    outerArray[i].free();
  }

  outerArray.free();
  assert_empty_map(true);
}
#endif  // CHAI_DISABLE_RM
#endif  // defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

TEST(ManagedArray, Clone)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<float> copy = array.clone();
  ASSERT_EQ(copy.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) { array[i] = -5.5 * i; });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(copy[i], i);
    ASSERT_EQ(array[i], -5.5 * i);
  });

  array.free();
  copy.free();
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
GPU_TEST(ManagedArray, DeviceClone)
{
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10u);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<float> copy = array.clone();
  ASSERT_EQ(copy.size(), 10u);

  forall(gpu(), 0, 10, [=] __device__(int i) { array[i] = -5.5 * i; });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(copy[i], i);
    ASSERT_EQ(array[i], -5.5 * i);
  });

  array.free();
  copy.free();
  assert_empty_map(true);
}

#ifndef CHAI_DISABLE_RM
GPU_TEST(ManagedArray, CopyConstruct)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
    array[i] = expectedValue;
  });

  chai::ManagedArray<int> array2 = array;

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = array2[i];
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  array.free();
  results.free();
  assert_empty_map(true);
}

#endif
#endif

TEST(ManagedArray, SizeZero)
{
  chai::ManagedArray<double> array;
  ASSERT_EQ(array.size(), 0u);
  array.allocate(0);
  ASSERT_EQ(array.size(), 0u);
  assert_empty_map(true);
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
GPU_TEST(ManagedArray, CopyZero)
{
  chai::ManagedArray<double> array;
  array.allocate(0);
  ASSERT_EQ(array.size(), 0u);

  forall(gpu(), 0, 1, [=] __device__ (int) {
    (void) array;
  });

  forall(sequential(), 0, 1, [=] (int) { 
    (void) array;
  });

  array.free();
  assert_empty_map(true);
}
#endif

TEST(ManagedArray, NoAllocation)
{
  chai::ManagedArray<double> array(10, chai::NONE);
  double* data = array.data(chai::NONE, false);
  ASSERT_EQ(data, nullptr);

  forall(sequential(), 0, 10, [=] (int i) {
    array[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

TEST(ManagedArray, NoAllocationNull)
{
  chai::ManagedArray<double> array;
  array.allocate(10, chai::NONE);

  forall(sequential(), 0, 10, [=] (int i) {
    array[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
GPU_TEST(ManagedArray, NoAllocationGPU)
{
  chai::ManagedArray<double> array(10, chai::NONE);

  forall(gpu(), 0, 10, [=] __device__ (int i) {
    array[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

GPU_TEST(ManagedArray, NoAllocationGPUList)
{
  auto& rm = umpire::ResourceManager::getInstance();
  chai::ManagedArray<double> array(10,
      std::initializer_list<chai::ExecutionSpace>{chai::CPU},
      std::initializer_list<umpire::Allocator>{rm.getAllocator("HOST")}
  );

  forall(gpu(), 0, 10, [=] __device__ (int i) {
    array[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

GPU_TEST(ManagedArray, NoAllocationNullGPU)
{
  chai::ManagedArray<double> array;
  array.allocate(10, chai::NONE);

  forall(gpu(), 0, 10, [=] __device__ (int i) {
    array[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}
#endif
