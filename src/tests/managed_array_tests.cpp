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
  ASSERT_EQ(array.size(), 0u);
}

TEST(ManagedArray, SizeConstructor) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

TEST(ManagedArray, SpaceConstructorCPU) {
  chai::ManagedArray<float> array(10, chai::CPU);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

#if defined(CHAI_ENABLE_CUDA)
TEST(ManagedArray, SpaceConstructorGPU) {
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.size(), 10u);
  array.free();
}

#if defined(CHAI_ENABLE_UM)
TEST(ManagedArray, SpaceConstructorUM) {
  chai::ManagedArray<float> array(10, chai::UM);
  ASSERT_EQ(array.size(), 10u);
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

  forall(sequential(), 0, 10, [=] (int i) {
      ASSERT_EQ(array_const[i], i);
  });
}

#if defined(CHAI_ENABLE_CUDA)
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
  ASSERT_EQ(array.size(), 0u);

  array.allocate(10);
  ASSERT_EQ(array.size(), 10u);
}

TEST(ManagedArray, ReallocateCPU) {
  chai::ManagedArray<float> array(10);
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
CUDA_TEST(ManagedArray, ReallocateGPU) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.size(), 10u);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  array.reallocate(20);
  ASSERT_EQ(array.size(), 20u);

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

  ASSERT_EQ(a.size(), 0u);
  ASSERT_EQ(b.size(), 0u);

  chai::ManagedArray<float> c(nullptr);

  ASSERT_EQ(c.size(), 0u);
}

#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
TEST(ManagedArray, ImplicitConversions) {
  chai::ManagedArray<float> a(10);

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

#if defined(CHAI_ENABLE_CUDA)
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

TEST(ManagedArray, ExternalConstructorUnowned) {
  float* data = new float[100];

  for (int i = 0; i < 100; i++) {
    data[i] = 1.0f*i;
  }

  chai::ManagedArray<float> array = chai::makeManagedArray<float>(data, 100, chai::CPU, false);

  forall(sequential(), 0, 20, [=] (int i) {
    ASSERT_EQ(data[i], array[i]);
  });

  array.free();

  ASSERT_NE(nullptr, data);
}

TEST(ManagedArray, ExternalConstructorOwned) {
  float* data = new float[20];

  for (int i = 0; i < 20; i++) {
    data[i] = 1.0f*i;
  }

  chai::ManagedArray<float> array = chai::makeManagedArray<float>(data, 20, chai::CPU, true);

  forall(sequential(), 0, 20, [=] (int i) {
    ASSERT_EQ(data[i], array[i]);
  });

  array.free();
}

TEST(ManagedArray, Reset)
{
  chai::ManagedArray<float> array(20);

  forall(sequential(), 0, 20, [=] (int i) {
    array[i] = 1.0f*i;
  });

  array.reset();
}

#if defined(CHAI_ENABLE_CUDA)
#ifndef CHAI_DISABLE_RM
CUDA_TEST(ManagedArray, ResetDevice)
{
  chai::ManagedArray<float> array(20);

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
  array.allocate(20, chai::CPU, [&](chai::Action act, chai::ExecutionSpace s, size_t bytes){  
	  //printf("cback: act=%d, space=%d, bytes=%ld\n",
		//   (int)act, (int)s, (long)bytes);
		if(act == chai::ACTION_MOVE){
	    if(s == chai::CPU){ ++num_d2h; bytes_d2h += bytes;}
  	  if(s == chai::GPU){ ++num_h2d; bytes_h2d += bytes;}
	  }
		if(act == chai::ACTION_ALLOC){
			bytes_alloc += bytes;	
		}
		if(act == chai::ACTION_FREE){
			bytes_free += bytes;	
		}
  });

  for(int iter = 0;iter < 10;++ iter){
    forall(sequential(), 0, 20, [=] (int i) {
      array[i] = 0.0f;
    });

    forall(cuda(), 0, 20, [=] __device__ (int i) {
      array[i] = 1.0f*i;
    });
  }
  

  ASSERT_EQ(num_d2h, 9);
  ASSERT_EQ(bytes_d2h, 9*sizeof(float)*20);
  ASSERT_EQ(num_h2d, 10);
  ASSERT_EQ(bytes_h2d, 10*sizeof(float)*20);  
	
	array.free();

	ASSERT_EQ(bytes_alloc, 2*20*sizeof(float));
	ASSERT_EQ(bytes_free, 2*20*sizeof(float));
}
#endif
#endif

#if !defined(NDEBUG)
TEST(ManagedArray, OutOfRangeAccess)
{
  chai::ManagedArray<float> array(20);

  EXPECT_DEATH(
  forall(sequential(), 10, 50, [=] (int i) {
    array[i] = 0.0f;
  });,
  "i < m_elems");
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ManagedArray, OutOfRangeAccessGPU)
{
  chai::ManagedArray<float> array(20);

  forall(cuda(), 10, 50, [=] __device__ (int i) {
    array[i] = 0.0f;
  });
}
#endif // defined(CHAI_ENABLE_CUDA)
#endif // !defined(NDEBUG)
