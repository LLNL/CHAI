// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
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

#include "chai/config.hpp"

#include "../src/util/forall.hpp"

#include "chai/ManagedArray.hpp"


struct my_point {
  double x;
  double y;
};

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

#if defined(CHAI_ENABLE_CUDA)
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

TEST(ManagedArray, SetOnHost)
{
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

  array.free();
}

TEST(ManagedArray, Const)
{
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const float> array_const(array);

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array_const[i], i); });
}

#if defined(CHAI_ENABLE_PICK)
TEST(ManagedArray, PickHostFromHostConst)
{
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

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
TEST(ManagedArray, PickHostFromHostConstUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(sequential(), 0, 10, [=](int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  // array_const.free();
}

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

CUDA_TEST(ManagedArray, PickHostFromDeviceConstUM)
{
  chai::ManagedArray<int> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  chai::ManagedArray<const int> array_const(array);

  int temp = array_const.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
  // array_const.free();
}

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

CUDA_TEST(ManagedArray, SetOnDevice)
{
  chai::ManagedArray<float> array(10);

  forall(cuda(), 0, 10, [=] __device__(int i) { array[i] = i; });

  forall(sequential(), 0, 10, [=](int i) { ASSERT_EQ(array[i], i); });

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
}
#endif

TEST(ManagedArray, NullpointerConversions)
{
  chai::ManagedArray<float> a;
  a = nullptr;

  chai::ManagedArray<const float> b;
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
}
#endif

#ifndef CHAI_DISABLE_RM
TEST(ManagedArray, ExternalConstructorUnowned)
{
  float* data = new float[100];

  for (int i = 0; i < 100; i++) {
    data[i] = 1.0f * i;
  }

  chai::ManagedArray<float> array =
      chai::makeManagedArray<float>(data, 100, chai::CPU, false);

  forall(sequential(), 0, 20, [=](int i) { ASSERT_EQ(data[i], array[i]); });

  array.free();

  ASSERT_NE(nullptr, data);
}

TEST(ManagedArray, ExternalConstructorOwned)
{
  float* data = new float[20];

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
                   // printf("cback: act=%d, space=%d, bytes=%ld\n",
                   //   (int)act, (int)s, (long)bytes);
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

CUDA_TEST(ManagedArray, MoveInnerImpl)
{
  chai::ManagedArray<chai::ManagedArray<int>> originalArray(3, chai::CPU);

  for (int i = 0; i < 3; ++i) {
    auto temp = chai::ManagedArray<int>(5, chai::GPU);

    forall(cuda(), 0, 5, [=] __device__(int j) { temp[j] = j; });

    originalArray[i] = temp;
  }

  auto copiedArray = chai::ManagedArray<chai::ManagedArray<int>>(originalArray);

  for (int i = 0; i < 3; ++i) {
    auto temp = copiedArray[i];

    forall(sequential(), 0, 5, [=](int j) { ASSERT_EQ(temp[j], j); });
  }

  for (int i = 0; i < 3; ++i) {
    copiedArray[i].free();
  }

  copiedArray.free();
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
