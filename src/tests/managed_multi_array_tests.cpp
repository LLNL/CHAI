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

#include "chai/ManagedMultiArray.hpp"

#include "chai/config.hpp"

struct my_point {
  double x;
  double y;
};

TEST(ManagedMultiArray, DefaultConstructor) {
  chai::ManagedMultiArray<float,1> array;
  ASSERT_EQ(array.size(), 0u);
}

TEST(ManagedMultiArray, SizeConstructor_1D) {
  chai::ManagedMultiArray<float,1> array(10);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

TEST(ManagedMultiArray, SizeConstructor_2D) {
  chai::ManagedMultiArray<float,2> array(10, 2 );
  ASSERT_EQ(array.size(), 20u);
  array.free();
}

TEST(ManagedMultiArray, SizeConstructor_3D) {
  chai::ManagedMultiArray<float,3> array(10, 2, 3 );
  ASSERT_EQ(array.size(), 60u);
  array.free();
}

TEST(ManagedMultiArray, SpaceConstructorCPU_1D) {
  chai::ManagedMultiArray<float,1> array(chai::CPU,10);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

TEST(ManagedMultiArray, SpaceConstructorCPU_2D) {
  chai::ManagedMultiArray<float,2> array(chai::CPU,10,2);
  ASSERT_EQ(array.size(), 20u);
  array.free();
}

TEST(ManagedMultiArray, SpaceConstructorCPU_3D) {
  chai::ManagedMultiArray<float,3> array(chai::CPU,10,2,3);
  ASSERT_EQ(array.size(), 60u);
  array.free();
}


#if defined(CHAI_ENABLE_CUDA)
TEST(ManagedMultiArray, SpaceConstructorGPU) {
  chai::ManagedMultiArray<float,1> array(chai::GPU,10);
  ASSERT_EQ(array.size(), 10);
  array.free();
}

#if defined(CHAI_ENABLE_UM)
TEST(ManagedMultiArray, SpaceConstructorUM) {
  chai::ManagedMultiArray<float,1> array(chai::UM,10);
  ASSERT_EQ(array.size(), 10);
  array.free();
}
#endif
#endif

TEST(ManagedMultiArray, SetOnHost) {
  chai::ManagedMultiArray<float,1> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

//TEST(ManagedMultiArray, Const) {
//  chai::ManagedMultiArray<float> array(10);
//
//  forall(sequential(), 0, 10, [=] (int i) {
//      array[i] = i;
//  });
//
//  chai::ManagedMultiArray<const float> array_const(array);
////  chai::ManagedMultiArray<const float> array_const2 = array;
//
//  forall(sequential(), 0, 10, [=] (int i) {
//      ASSERT_EQ(array_const[i], i);
//  });
//}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedMultiArray, SetOnDevice) {
  chai::ManagedMultiArray<float,1> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

CUDA_TEST(ManagedMultiArray, GetGpuOnHost) {
  chai::ManagedMultiArray<float,1> array(chai::GPU, 10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i);
  });

  array.free();
}

#if defined(CHAI_ENABLE_UM)
CUDA_TEST(ManagedMultiArray, SetOnDeviceUM) {
  chai::ManagedMultiArray<float,1> array(chai::UM, 10);

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

TEST(ManagedMultiArray, Allocate) {
  chai::ManagedMultiArray<float,1> array;
  ASSERT_EQ(array.size(), 0u);

  array.allocate(10);
  ASSERT_EQ(array.size(), 10u);
}

TEST(ManagedMultiArray, ReallocateCPU) {
  chai::ManagedMultiArray<float,1> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(sequential(), 0, 10, [=](int i) {
      array[i] = i;
  });

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
CUDA_TEST(ManagedMultiArray, ReallocateGPU) {
  chai::ManagedMultiArray<float,1> array(10);
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

//TEST(ManagedMultiArray, NullpointerConversions) {
//  chai::ManagedMultiArray<float> a;
//  a = nullptr;
//
//  chai::ManagedMultiArray<const float> b;
//  b = nullptr;
//
//  ASSERT_EQ(a.size(), 0);
//  ASSERT_EQ(b.size(), 0);
//
//  chai::ManagedMultiArray<float> c(nullptr);
//
//  ASSERT_EQ(c.size(), 0);
//}
//
//#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
//TEST(ManagedMultiArray, ImplicitConversions) {
//  chai::ManagedMultiArray<float> a(10);
//  float * raw_a = a;
//
//  chai::ManagedMultiArray<float> a2 = a;
//
//  SUCCEED();
//}
//#endif

TEST(ManagedMultiArray, PodTest) {
  chai::ManagedMultiArray<my_point,1> array(1);

  forall(sequential(), 0, 1, [=] (int i) {
      array[i].x = (double) i;
      array[i].y = (double) i*2.0;
  });

  forall(sequential(), 0, 1, [=] (int i) {
    ASSERT_EQ(array[i].x, i);
    ASSERT_EQ(array[i].y, i*2.0);
  });
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedMultiArray, PodTestGPU) {
  chai::ManagedMultiArray<my_point,1> array(1);

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

//TEST(ManagedMultiArray, ExternalConstructorUnowned) {
//  float* data = new float[100];
//
//  for (int i = 0; i < 100; i++) {
//    data[i] = 1.0f*i;
//  }
//
//  chai::ManagedMultiArray<float> array = chai::makeManagedMultiArray<float>(data, 100, chai::CPU, false);
//
//  forall(sequential(), 0, 20, [=] (int i) {
//    ASSERT_EQ(data[i], array[i]);
//  });
//
//  array.free();
//
//  ASSERT_NE(nullptr, data);
//}
//
//TEST(ManagedMultiArray, ExternalConstructorOwned) {
//  float* data = new float[20];
//
//  for (int i = 0; i < 20; i++) {
//    data[i] = 1.0f*i;
//  }
//
//  chai::ManagedMultiArray<float> array = chai::makeManagedMultiArray<float>(data, 20, chai::CPU, true);
//
//  forall(sequential(), 0, 20, [=] (int i) {
//    ASSERT_EQ(data[i], array[i]);
//  });
//
//  array.free();
//}

TEST(ManagedMultiArray, Reset)
{
  chai::ManagedMultiArray<float,1> array(20);

  forall(sequential(), 0, 20, [=] (int i) {
    array[i] = 1.0f*i;
  });

  array.reset();
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedMultiArray, ResetDevice)
{
  chai::ManagedMultiArray<float,1> array(20);

  forall(sequential(), 0, 20, [=] (int i) {
    array[i] = 0.0f;
  });

  forall(cuda(), 0, 20, [=] __device__ (int i) {
    array[i] = 1.0f*i;
  });

  array.reset();

  forall(sequential(), 0, 20, [=] (int i) {
    ASSERT_EQ(array[i], 0.0f);
  });
}
#endif
