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

#if defined(ENABLE_CUDA)
TEST(ManagedArray, SpaceConstructorGPU) {
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10);
  array.free();
}
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

#if defined(ENABLE_CUDA)
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

#if defined(ENABLE_CUDA)
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

#if defined(ENABLE_IMPLICIT_CONVERSIONS)
TEST(ManageArray, ImplicitConversions) {
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
TEST(ManagedArray, PodTestGPU) {
  chai::ManagedArray<my_point> array(1);

  forall(cuda(), 0, 1, [=] (int i) {
      array[i].x = (double) i;
      array[i].y = (double) i*2.0;
  });

  forall(sequential(), 0, 1, [=] (int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i*2.0);
  });
}

#endif
