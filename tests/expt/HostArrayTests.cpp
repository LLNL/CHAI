#include "gtest/gtest.h"
#include "chai/expt/HostArray.hpp"
#include "umpire/ResourceManager.hpp"

TEST(HostArrayTest, DefaultConstructor) {
  chai::expt::HostArray<int> array;
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.data(), nullptr);
}

TEST(HostArrayTest, SizeConstructor) {
  const size_t testSize = 10;
  chai::expt::HostArray<double> array(testSize);
  
  EXPECT_EQ(array.size(), testSize);
  EXPECT_NE(array.data(), nullptr);
}

TEST(HostArrayTest, AllocatorConstructor) {
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  
  chai::expt::HostArray<float> array(allocator);
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.data(), nullptr);
}

TEST(HostArrayTest, SizeAllocatorConstructor) {
  const size_t testSize = 5;
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");
  
  chai::expt::HostArray<int> array(testSize, allocator);
  
  EXPECT_EQ(array.size(), testSize);
  EXPECT_NE(array.data(), nullptr);
}

TEST(HostArrayTest, CopyConstructor) {
  const size_t testSize = 3;
  chai::expt::HostArray<int> array1(testSize);
  
  // Initialize array1
  for (size_t i = 0; i < testSize; ++i) {
    array1[i] = static_cast<int>(i * 10);
  }
  
  // Copy construct array2
  chai::expt::HostArray<int> array2(array1);
  
  EXPECT_EQ(array2.size(), array1.size());
  
  // Verify contents
  for (size_t i = 0; i < testSize; ++i) {
    EXPECT_EQ(array2[i], array1[i]);
  }
  
  // Verify array2 is a deep copy
  array1[0] = 100;
  EXPECT_NE(array1[0], array2[0]);
}

TEST(HostArrayTest, MoveConstructor) {
  const size_t testSize = 4;
  chai::expt::HostArray<double> array1(testSize);
  
  // Initialize array1
  for (size_t i = 0; i < testSize; ++i) {
    array1[i] = i * 1.5;
  }
  
  double* originalData = array1.data();
  
  // Move construct array2
  chai::expt::HostArray<double> array2(std::move(array1));
  
  // Verify array1 is empty after move
  EXPECT_EQ(array1.size(), 0);
  EXPECT_EQ(array1.data(), nullptr);
  
  // Verify array2 has the original data
  EXPECT_EQ(array2.size(), testSize);
  EXPECT_EQ(array2.data(), originalData);
  EXPECT_DOUBLE_EQ(array2[2], 3.0);
}

TEST(HostArrayTest, CopyAssignment) {
  const size_t srcSize = 5;
  const size_t destSize = 3;
  
  chai::expt::HostArray<int> array1(srcSize);
  chai::expt::HostArray<int> array2(destSize);
  
  // Initialize arrays
  for (size_t i = 0; i < srcSize; ++i) {
    array1[i] = static_cast<int>(i * 10);
  }
  
  for (size_t i = 0; i < destSize; ++i) {
    array2[i] = static_cast<int>(i * 100);
  }
  
  // Perform copy assignment
  array2 = array1;
  
  // Verify array2 now matches array1
  EXPECT_EQ(array2.size(), array1.size());
  
  for (size_t i = 0; i < srcSize; ++i) {
    EXPECT_EQ(array2[i], array1[i]);
  }
}

TEST(HostArrayTest, MoveAssignment) {
  const size_t srcSize = 4;
  const size_t destSize = 2;
  
  chai::expt::HostArray<float> array1(srcSize);
  chai::expt::HostArray<float> array2(destSize);
  
  // Initialize arrays
  for (size_t i = 0; i < srcSize; ++i) {
    array1[i] = static_cast<float>(i * 2.5);
  }
  
  float* originalData = array1.data();
  
  // Perform move assignment
  array2 = std::move(array1);
  
  // Verify array1 is empty after move
  EXPECT_EQ(array1.size(), 0);
  EXPECT_EQ(array1.data(), nullptr);
  
  // Verify array2 now has array1's original data
  EXPECT_EQ(array2.size(), srcSize);
  EXPECT_EQ(array2.data(), originalData);
}

TEST(HostArrayTest, Resize) {
  const size_t initialSize = 3;
  chai::expt::HostArray<int> array(initialSize);
  
  // Initialize array
  for (size_t i = 0; i < initialSize; ++i) {
    array[i] = static_cast<int>(i + 1);
  }
  
  // Resize larger
  const size_t newLargerSize = 5;
  array.resize(newLargerSize);
  
  EXPECT_EQ(array.size(), newLargerSize);
  
  // Check original data was preserved
  for (size_t i = 0; i < initialSize; ++i) {
    EXPECT_EQ(array[i], static_cast<int>(i + 1));
  }
  
  // Resize smaller
  const size_t newSmallerSize = 2;
  array.resize(newSmallerSize);
  
  EXPECT_EQ(array.size(), newSmallerSize);
  
  // Check remaining data was preserved
  for (size_t i = 0; i < newSmallerSize; ++i) {
    EXPECT_EQ(array[i], static_cast<int>(i + 1));
  }
}

TEST(HostArrayTest, Free) {
  chai::expt::HostArray<double> array(10);
  EXPECT_NE(array.data(), nullptr);
  
  array.free();
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.data(), nullptr);
}

TEST(HostArrayTest, AccessOperators) {
  const size_t testSize = 4;
  chai::expt::HostArray<int> array(testSize);
  
  // Test set and operator[]
  for (size_t i = 0; i < testSize; ++i) {
    array[i] = static_cast<int>(i * 10);
  }
  
  // Test get and const operator[]
  const chai::expt::HostArray<int>& constArray = array;
  for (size_t i = 0; i < testSize; ++i) {
    EXPECT_EQ(constArray[i], static_cast<int>(i * 10));
    EXPECT_EQ(constArray.get(i), static_cast<int>(i * 10));
  }
  
  // Test set method
  for (size_t i = 0; i < testSize; ++i) {
    array.set(i, static_cast<int>(i * 20));
    EXPECT_EQ(array[i], static_cast<int>(i * 20));
  }
}

TEST(HostArrayTest, NonTrivialType) {
  // Test with a non-trivial type like std::string
  const size_t testSize = 3;
  chai::expt::HostArray<std::string> array(testSize);
  
  array[0] = "Hello";
  array[1] = "World";
  array[2] = "Test";
  
  // Copy construction
  chai::expt::HostArray<std::string> arrayCopy(array);
  EXPECT_EQ(arrayCopy.size(), testSize);
  EXPECT_EQ(arrayCopy[0], "Hello");
  EXPECT_EQ(arrayCopy[1], "World");
  EXPECT_EQ(arrayCopy[2], "Test");
  
  // Modify original, copy should be unaffected
  array[0] = "Changed";
  EXPECT_EQ(arrayCopy[0], "Hello");
  
  // Resize
  arrayCopy.resize(4);
  EXPECT_EQ(arrayCopy.size(), 4);
  EXPECT_EQ(arrayCopy[0], "Hello");
  arrayCopy[3] = "New";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}