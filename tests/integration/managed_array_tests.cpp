// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
// rights reserved.
//
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of CHAI.
//
// LLNL-CODE-705877
//
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()

#ifdef NDEBUG
#define device_assert(EXP) if( !EXP ) asm ("trap;")
#else
#define device_assert(EXP) assert(EXP)
#endif

#include "chai/config.hpp"

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"


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
}

#if (!defined(CHAI_DISABLE_RM))
TEST(ManagedArray, Const)
{
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const float> array_const(array);

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array_const[i], i); });

  array.free();
}
#endif

TEST(ManagedArray, Slice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl = array.slice(0,5);
  sl.free();
  array.free();
}

TEST(ManagedArray, SliceOfSlice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl1 = array.slice(0,6);
  chai::ManagedArray<float> sl2 = sl1.slice(3,3);

  forall(sequential(), 0, 3, [=] (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 3, [=] (int i) {
    ASSERT_EQ(array[i], array[i+3]);
  });

  sl1.free();
  sl2.free();
  array.free();
}

#if defined(CHAI_ENABLE_PICK)
#if (!defined(CHAI_DISABLE_RM))
TEST(ManagedArray, PickHostFromHostConst) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}
#endif

TEST(ManagedArray, PickHostFromHost)
{
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

TEST(ManagedArray, SetHostToHost)
{
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  ASSERT_EQ(array[5], 10);

  array.free();
}


TEST(ManagedArray, IncrementDecrementOnHost)
{
  chai::ManagedArray<int> arrayI(10);
  chai::ManagedArray<int> arrayD(10);

  forall(sequential(), 0, 10, [=](int i) {
    arrayI[i] = i;
    arrayD[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) {
    arrayI.incr(i);
    arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(arrayI[i], i + 1);
    ASSERT_EQ(arrayD[i], i - 1);
  });

  arrayI.free();
  arrayD.free();
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
  // array_const.free();
}
#endif

TEST(ManagedArray, PickHostFromHostUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

TEST(ManagedArray, SetHostToHostUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  ASSERT_EQ(array[5], 10);

  array.free();
}

TEST(ManagedArray, IncrementDecrementOnHostUM)
{
  chai::ManagedArray<int> arrayI(10, chai::UM);
  chai::ManagedArray<int> arrayD(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) {
    arrayI[i] = i;
    arrayD[i] = i;
  });

  forall(sequential(), 0, 10, [=](int i) {
    arrayI.incr(i);
    arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(arrayI[i], i + 1);
    ASSERT_EQ(arrayD[i], i - 1);
  });

  arrayI.free();
  arrayD.free();
}
#endif

#endif

#if defined(CHAI_ENABLE_CUDA)
#if defined(CHAI_ENABLE_PICK)

#if defined(CHAI_ENABLE_UM)
CUDA_TEST(ManagedArray, PickandSetDeviceToDeviceUM)
{
  chai::ManagedArray<int> array1(10, chai::UM);
  chai::ManagedArray<int> array2(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array1[i] = i; });

  forall(cuda(), 0, 10, [=] __device__(int i) {
    int temp = array1.pick(i);
    array2.set(i, temp);
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array2[i], i); });

  array1.free();
  array2.free();
}

CUDA_TEST(ManagedArray, PickHostFromDeviceUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

#if (!defined(CHAI_DISABLE_RM))
CUDA_TEST(ManagedArray, PickHostFromDeviceConstUM) {
  chai::ManagedArray<int> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  // array_const.free();
}
#endif

CUDA_TEST(ManagedArray, SetHostToDeviceUM)
{
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  temp = array.pick(5);
  ASSERT_EQ(temp, 10);

  array.free();
}

CUDA_TEST(ManagedArray, IncrementDecrementOnDeviceUM)
{
  chai::ManagedArray<int> arrayI(10, chai::UM);
  chai::ManagedArray<int> arrayD(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) {
    arrayI[i] = i;
    arrayD[i] = i;
  });

  forall(cuda(), 0, 10, [=] __device__(int i) {
    arrayI.incr(i);
    arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(arrayI[i], i + 1);
    ASSERT_EQ(arrayD[i], i - 1);
  });

  arrayI.free();
  arrayD.free();
}

CUDA_TEST(ManagedArray, IncrementDecrementFromHostOnDeviceUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  array.incr(5);
  array.decr(9);

  int temp;
  temp = array.pick(5);
  ASSERT_EQ(temp, 6);

  temp = array.pick(9);
  ASSERT_EQ(temp, 8);

  array.free();
}

CUDA_TEST(ManagedArray, PickandSetSliceDeviceToDeviceUM) {
  chai::ManagedArray<int> array(10, chai::UM);
  chai::ManagedArray<int> sl1 = array.slice(0,5);
  chai::ManagedArray<int> sl2 = array.slice(5,5);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(cuda(), 0, 5, [=] __device__ (int i) {
      int temp = sl2.pick(i);
      temp += sl2.pick(i);
      sl1.set(i, temp);
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], (i+5)*2);
  });

  array.free();
}
#endif

#if (!defined(CHAI_DISABLE_RM))
CUDA_TEST(ManagedArray, PickandSetDeviceToDevice)
{
  chai::ManagedArray<int> array1(10);
  chai::ManagedArray<int> array2(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array1[i] = i; });

  chai::ManagedArray<const int> array_const(array1);

  forall(cuda(), 0, 10, [=] __device__(int i) {
    int temp = array1.pick(i);
    temp += array_const.pick(i);
    array2.set(i, temp);
  });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array2[i], i + i); });

  array1.free();
  array2.free();
}

CUDA_TEST(ManagedArray, PickandSetSliceDeviceToDevice) {
  chai::ManagedArray<int> array(10);
  chai::ManagedArray<int> sl1 = array.slice(0,5);
  chai::ManagedArray<int> sl2 = array.slice(5,5);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(cuda(), 0, 5, [=] __device__ (int i) {
      int temp = sl2.pick(i);
      temp += sl2.pick(i);
      sl1.set(i, temp);
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], (i+5)*2);
  });

  array.free();
}


CUDA_TEST(ManagedArray, PickHostFromDevice)
{
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

CUDA_TEST(ManagedArray, PickHostFromDeviceConst)
{
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array_const[i], i); });

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  // array_const.free();
}

CUDA_TEST(ManagedArray, SetHostToDevice)
{
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  int temp = 10;
  array.set(5, temp);
  temp = array.pick(5);
  ASSERT_EQ(temp, 10);

  array.free();
}
CUDA_TEST(ManagedArray, IncrementDecrementOnDevice)
{
  chai::ManagedArray<int> arrayI(10);
  chai::ManagedArray<int> arrayD(10);

  forall(cuda(), 0, 10, [=] __device__(int i) {
    arrayI[i] = i;
    arrayD[i] = i;
  });

  forall(cuda(), 0, 10, [=] __device__(int i) {
    arrayI.incr(i);
    arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(arrayI[i], i + 1);
    ASSERT_EQ(arrayD[i], i - 1);
  });

  arrayI.free();
  arrayD.free();
}

CUDA_TEST(ManagedArray, IncrementDecrementFromHostOnDevice)
{
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  array.incr(5);
  array.decr(9);

  int temp;
  temp = array.pick(5);
  ASSERT_EQ(temp, 6);

  temp = array.pick(9);
  ASSERT_EQ(temp, 8);

  array.free();
}
#endif
#endif

CUDA_TEST(ManagedArray, SliceOfSliceDevice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl1 = array.slice(0,6);
  chai::ManagedArray<float> sl2 = sl1.slice(3,3);

  forall(cuda(), 0, 3, [=] __device__ (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 3, [=] (int i) {
    ASSERT_EQ(array[i], array[i+3]);
  });

  sl1.free();
  sl2.free();
  array.free();
}

CUDA_TEST(ManagedArray, SliceDevice) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<float> sl1 = array.slice(0,5);
  chai::ManagedArray<float> sl2 = array.slice(5,5);

  forall(cuda(), 0, 5, [=] __device__ (int i) {
      sl1[i] = sl2[i];
  });

  forall(sequential(), 0, 5, [=] (int i) {
    ASSERT_EQ(array[i], array[i+5]);
  });

  sl1.free();
  sl2.free();
  array.free();
}

CUDA_TEST(ManagedArray, SetOnDevice) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] *= 2; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], 2 * i); });

  array.free();
}

CUDA_TEST(ManagedArray, GetGpuOnHost)
{
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

#if defined(CHAI_ENABLE_UM)
CUDA_TEST(ManagedArray, SetOnDeviceUM)
{
  chai::ManagedArray<float> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
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
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedArray, ReallocateGPU)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

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
}

#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
TEST(ManagedArray, ImplicitConversions)
{
  chai::ManagedArray<float> a(10);

  chai::ManagedArray<float> a2 = a;

  a.free();
  SUCCEED();
}
#endif

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
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedArray, PodTestGPU)
{
  chai::ManagedArray<my_point> array(1);

  forall(cuda(), 0, 1, [=] __device__(int i) {
    array[i].x = (double)i;
    array[i].y = (double)i * 2.0;
  });

  forall(sequential(), 0, 1, [=](int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i * 2.0);
  });

  array.free();
}
#endif

#ifndef CHAI_DISABLE_RM
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
}
#endif

TEST(ManagedArray, Reset)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 1.0f * i; });

  array.reset();
  array.free();
}

#if defined(CHAI_ENABLE_CUDA)
#ifndef CHAI_DISABLE_RM
CUDA_TEST(ManagedArray, ResetDevice)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=](int i) { array[i] = 0.0f; });

  forall(cuda(), 0, 20, [=] __device__(int i) { array[i] = 1.0f * i; });

  array.reset();

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(array[i], 0.0f); });

  array.free();
}
#endif
#endif


#if defined(CHAI_ENABLE_CUDA)
#ifndef CHAI_DISABLE_RM
CUDA_TEST(ManagedArray, UserCallback)
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
                 [&](chai::Action act, chai::ExecutionSpace s, size_t bytes) {
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

    forall(cuda(), 0, 20, [=] __device__(int i) { array[i] = 1.0f * i; });
  }


  ASSERT_EQ(num_d2h, 9);
  ASSERT_EQ(bytes_d2h, 9 * sizeof(float) * 20);
  ASSERT_EQ(num_h2d, 10);
  ASSERT_EQ(bytes_h2d, 10 * sizeof(float) * 20);

  array.free();

  ASSERT_EQ(bytes_alloc, 2 * 20 * sizeof(float));
  ASSERT_EQ(bytes_free, 2 * 20 * sizeof(float));
}

CUDA_TEST(ManagedArray, CallBackConst)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&](chai::Action act, chai::ExecutionSpace s, size_t bytes)
  {
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

  chai::ManagedArray<int> array(100);
  array.setUserCallback(callBack);

  // Set the values.
  forall(sequential(), 0, 100, [=](int i) { array[i] = i; });

  // Create a const copy and move it to the device.
  chai::ManagedArray<int const> array_const = array;
  forall(cuda(), 0, 100, [=] __device__(int i) { device_assert(array_const[i] == i); });

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
}

CUDA_TEST(ManagedArray, CallBackConstArray)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&](chai::Action act, chai::ExecutionSpace s, size_t bytes)
  {
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

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);
      temp.setUserCallback(callBack);

      forall(sequential(), 0, N,
        [=](int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
    }
  );

  // Create a const copy and move it to the device.
  chai::ManagedArray<chai::ManagedArray<int> const> outerArrayConst = outerArray;
  forall(cuda(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        device_assert(outerArrayConst[i][j] == N * i + j);
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
  }

  outerArray.free();
}

CUDA_TEST(ManagedArray, CallBackConstArrayConst)
{
  int num_h2d = 0;
  int num_d2h = 0;

  auto callBack = [&](chai::Action act, chai::ExecutionSpace s, size_t bytes)
  {
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

  /* Loop over the outer array and populate it with arrays on the CPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N);
      temp.setUserCallback(callBack);

      forall(sequential(), 0, N,
        [=](int j)
        {
          temp[j] = N * i + j;
        }
      );

      outerArray[i] = temp;
    }
  );

  // Create a const copy of int const and move it to the device.
  chai::ManagedArray<chai::ManagedArray<int const> const> outerArrayConst = 
    reinterpret_cast<chai::ManagedArray<chai::ManagedArray<int const> const> &>(outerArray);
  forall(cuda(), 0, N,
    [=] __device__(int i)
    {
      for( int j = 0; j < N; ++j)
      {
        device_assert(outerArrayConst[i][j] == N * i + j);
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
  }

  outerArray.free();
}

#endif
#endif


#if defined(CHAI_ENABLE_CUDA)
#ifndef CHAI_DISABLE_RM
CUDA_TEST(ManagedArray, Move)
{
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  array.move(chai::CPU);

  ASSERT_EQ(array[5], 5);

  array.free();
}

/**
 * This test creates an array of arrays where the outer array is on the
 * CPU and the inner arrays are on the GPU. It then captures the outer array
 * on the CPU and checks that the inner arrays were moved to the CPU.
 */
CUDA_TEST(ManagedArray, MoveInnerToHost)
{
  const int N = 5;

  /* Create the outer array. */
  chai::ManagedArray<chai::ManagedArray<int>> outerArray(N);

  /* Loop over the outer array and populate it with arrays on the GPU. */
  forall(sequential(), 0, N,
    [=](int i)
    {
      chai::ManagedArray<int> temp(N, chai::GPU);

      forall(cuda(), 0, N,
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
}

/**
 * This test creates an array of arrays where both the outer array and inner
 * arrays are on the CPU. It then captures the outer array on the GPU and
 * modifies the values of the inner arrays. Finally it captures the outer array
 * on the CPU and checks that the values of the inner arrays were modified
 * correctly. 
 */
CUDA_TEST(ManagedArray, MoveInnerToDevice)
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
  forall(cuda(), 0, N,
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
}

/**
 * This test creates an array of arrays of arrays where all of the arrays begin
 * on the CPU. It then captures the outermost array on the GPU and modifies the
 * values of the innermost arrays. Finally it captures the outermost array
 * on the CPU and checks that the values of the innermost arrays were modified
 * correctly. 
 */
CUDA_TEST(ManagedArray, MoveInnerToDevice2)
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
  forall(cuda(), 0, N,
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
}

#endif  // CHAI_DISABLE_RM
#endif  // defined(CHAI_ENABLE_CUDA)


#ifndef CHAI_DISABLE_RM
TEST(ManagedArray, DeepCopy)
{
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<float> copy = chai::deepCopy(array);
  ASSERT_EQ(copy.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) { array[i] = -5.5 * i; });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(copy[i], i);
    ASSERT_EQ(array[i], -5.5 * i);
  });

  array.free();
  copy.free();
}
#endif

#if defined(CHAI_ENABLE_CUDA)
#ifndef CHAI_DISABLE_RM
CUDA_TEST(ManagedArray, DeviceDeepCopy)
{
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10u);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<float> copy = chai::deepCopy(array);
  ASSERT_EQ(copy.size(), 10u);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = -5.5 * i; });

  forall(sequential(), 0, 10, [=](int i) {
    ASSERT_EQ(copy[i], i);
    ASSERT_EQ(array[i], -5.5 * i);
  });

  array.free();
  copy.free();
}
#endif
#endif  // defined(CHAI_ENABLE_CUDA)

int main( int argc, char* argv[] )
{
  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  chai::ArrayManager::finalize();

  return result;
}
