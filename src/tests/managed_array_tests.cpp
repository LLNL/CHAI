#include "gtest/gtest.h"

#define CUDA_TEST(X, Y) \
static void cuda_test_ ## X ## Y();\
TEST(X,Y) { cuda_test_ ## X ## Y();}\
static void cuda_test_ ## X ## Y()

#include "chai/ManagedArray.hpp"

#include "../util/forall.hpp"

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
