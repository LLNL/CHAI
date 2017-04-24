#include "gtest/gtest.h"

#include "chai/ManagedArray.hpp"

#include "../util/forall.hpp"

TEST(ManagedArray, DefaultConstructor) {
  chai::ManagedArray<float> array;
  ASSERT_EQ(array.getSize(), 0);
}

TEST(ManagedArray, SizeConstructor) {
  chai::ManagedArray<float> array(10);
  ASSERT_EQ(array.getSize(), sizeof(float)*10);
}

TEST(ManagedArray, SpaceConstructor) {
  chai::ManagedArray<float> array(10, chai::CPU);
  ASSERT_EQ(array.getSize(), sizeof(float)*10);
}

TEST(ManagedArray, SpaceConstructor) {
  chai::ManagedArray<float> array(10, chai::GPU);
  ASSERT_EQ(array.getSize(), sizeof(float)*10);
}

TEST(ManagedArray, SetOnHost) {
  chai::ManagedArray<float> array(10);

  forall(sequential(), 0, 10, [=] (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i)*10);
  });
}

TEST(ManagedArray, SetOnDevice) {
  chai::ManagedArray<float> array(10);

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i)*10);
  });
}

TEST(ManagedArray, GetGpuOnHost) {
  chai::ManagedArray<float> array(10, chai::GPU);

  forall(cuda(), 0, 10, [=] __device_ (int i) {
      array[i] = i;
  });

  forall(sequential(), 0, 10, [=] (int i) {
    ASSERT_EQ(array[i], i)*10);
  });
}
