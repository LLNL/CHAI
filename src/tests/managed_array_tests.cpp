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

#define CUDA_TEST(X, Y) \
static void cuda_test_ ## X ## Y();\
TEST(X,Y) { cuda_test_ ## X ## Y();}\
static void cuda_test_ ## X ## Y()

#include "../util/forall.hpp"

#include "chai/ManagedArray.hpp"

#include "chai/config.hpp"

struct my_point {
  double x;
  double y;
};

TEST(ManagedArray, DefaultConstructor) {
  chai::ManagedArray<float> array;
  ASSERT_EQ(array.size(), 0);
}

TEST(ManagedArray, SizeConstructor) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10);
  array.free();
}

TEST(ManagedArray, SpaceConstructorCPU) {
  chai::ManagedArray<float> array(10, chai::CPU);
  ASSERT_EQ(array.size(), 10);
  array.free();
}

#if defined(CHAI_ENABLE_CUDA)
TEST(ManagedArray, SpaceConstructorGPU) {
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10);
  array.free();
}

#if defined(CHAI_ENABLE_UM)
TEST(ManagedArray, SpaceConstructorUM) {
  chai::ManagedArray<float> array(10, chai::UM);
  ASSERT_EQ(array.size(), 10);
  array.free();
}
#endif
#endif

TEST(ManagedArray, SetOnHost) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

TEST(ManagedArray, Const) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  chai::ManagedArray<const float> array_const(array);
  chai::ManagedArray<const float> array_const2 = array;

  forall(sequential(), 0, 10, [=] (int i) {
      ASSERT_EQ(array_const[i], i);
  });
}

#if defined(CHAI_ENABLE_PICK_SET_INCR_DECR)
TEST(ManagedArray, PickHostFromHost) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  int temp;
  array.pick(5, temp);
  ASSERT_EQ(temp, 5);

  array.free();
}

TEST(ManagedArray, PickReturnHostFromHost) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

TEST(ManagedArray, SetHostToHost) {
  chai::ManagedArray<int> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  int temp = 10;
  array.set(5, temp);
  ASSERT_EQ(array[5], 10);

  array.free();
}

TEST(ManagedArray, IncrementDecrementOnHost) {
  chai::ManagedArray<int> arrayI(10);
  chai::ManagedArray<int> arrayD(10);

  forall(sequential(), 0, 10, [=] (int i) {
      arrayI[i] = i;
      arrayD[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
      arrayI.incr(i);
      arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(arrayI[i], i+1);
    ASSERT_EQ(arrayD[i], i-1);
  });

  arrayI.free();
  arrayD.free();
}
#endif


#if defined(CHAI_ENABLE_CUDA)
#if defined(CHAI_ENABLE_PICK_SET_INCR_DECR)
CUDA_TEST(ManagedArray, PickandSetDeviceToDevice) {
  chai::ManagedArray<int> array1(10);
  chai::ManagedArray<int> array2(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array1[i] = i;
  });

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      int temp;
      array1.pick(i, temp);
      temp += array1.pick(i);
      array2.set(i, temp);
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array2[i], i+i);
  });

  array1.free();
  array2.free();
}

CUDA_TEST(ManagedArray, PickHostFromDevice) {
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  int temp;
  array.pick(5, temp);
  ASSERT_EQ(temp, 5);

  array.free();
}

CUDA_TEST(ManagedArray, PickReturnHostFromDevice) {
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  int temp = array.pick(5);
  ASSERT_EQ(temp, 5);

  array.free();
}

CUDA_TEST(ManagedArray, SetHostToDevice) {
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  int temp = 10;
  array.set(5, temp);
  array.pick(5, temp);
  ASSERT_EQ(temp, 10);

  array.free();
}

CUDA_TEST(ManagedArray, IncrementDecrementOnDevice) {
  chai::ManagedArray<int> arrayI(10);
  chai::ManagedArray<int> arrayD(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      arrayI[i] = i;
      arrayD[i] = i;
  });

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      arrayI.incr(i);
      arrayD.decr(i);
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(arrayI[i], i+1);
    ASSERT_EQ(arrayD[i], i-1);
  });

  arrayI.free();
  arrayD.free();
}

CUDA_TEST(ManagedArray, IncrementDecrementFromHostOnDevice) {
  chai::ManagedArray<int> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

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

CUDA_TEST(ManagedArray, SetOnDevice) {
  chai::ManagedArray<float> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

CUDA_TEST(ManagedArray, GetGpuOnHost) {
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

#if defined(CHAI_ENABLE_UM)
CUDA_TEST(ManagedArray, SetOnDeviceUM) {
  chai::ManagedArray<float> array(10, chai::UM);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}
#endif
#endif

TEST(ManagedArray, Allocate) {
  chai::ManagedArray<float> array;
  ASSERT_EQ(array.size(), 0);

  array.allocate(10);
  ASSERT_EQ(array.size(), 10);
}

TEST(ManagedArray, ReallocateCPU) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10);

  forall(sequential(), 0, 10, [=](int i) {
      array[i] = i;
  });

  array.reallocate(20);
  ASSERT_EQ(array.size(), 20);

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
CUDA_TEST(ManagedArray, ReallocateGPU) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  array.reallocate(20);
  ASSERT_EQ(array.size(), 20);

  forall(sequential(), 0, 20, [=] (int i) {
      if ( i < 10) {
      ASSERT_EQ(array[i], i);
      } else {
        array[i] = i;
        ASSERT_EQ(array[i], i);
      }
  });
}
#endif

TEST(ManagedArray, NullpointerConversions) {
  chai::ManagedArray<float> a;
  a = nullptr;

  chai::ManagedArray<const float> b;
  b = nullptr;

  ASSERT_EQ(a.size(), 0);
  ASSERT_EQ(b.size(), 0);

  chai::ManagedArray<float> c(nullptr);

  ASSERT_EQ(c.size(), 0);
}

#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
TEST(ManagedArray, ImplicitConversions) {
  chai::ManagedArray<float> a(10);
  float * raw_a = a;

  chai::ManagedArray<float> a2 = a;

  SUCCEED();
}
#endif

TEST(ManagedArray, PodTest) {
  chai::ManagedArray<my_point> array(1);

  forall(sequential(), 0, 1, [=] (int i) {
      array[i].x = (double) i;
      array[i].y = (double) i*2.0;
  });

  forall(sequential(), 0, 1, [=] (int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i*2.0);
  });
}

#if defined(ENABLE_CUDA)
CUDA_TEST(ManagedArray, PodTestGPU) {
  chai::ManagedArray<my_point> array(1);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
      array[i].x = (double) i;
      array[i].y = (double) i*2.0;
  });

  forall(sequential(), 0, 1, [=] (int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i*2.0);
  });
}

#endif
