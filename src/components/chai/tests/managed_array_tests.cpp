#include "gtest/gtest.h"

#include "ManagedArray.hpp"

TEST(ManagedArray, DefaultConstructor) {
  chai::ManagedArray<float> array;

  ASSERT_EQ(array.getSize(), 0);
}

TEST(ManagedArray, SizeConstructor) {
  chai::ManagedArray<float> array(10);

  ASSERT_EQ(array.getSize(), sizeof(float)*10);
}
