#include "gtest/gtest.h"
#include "chai/expt/CopyHidingArray.hpp"
#include <vector>

namespace chai {
namespace expt {

// Fixture class for CopyHidingArray tests
class CopyHidingArrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Get ResourceManager instance
    m_resource_manager = &umpire::ResourceManager::getInstance();
    
    // Get default allocators
    m_cpu_allocator = m_resource_manager->getAllocator("HOST");
    m_gpu_allocator = m_resource_manager->getAllocator("DEVICE");
  }

  umpire::ResourceManager* m_resource_manager;
  umpire::Allocator m_cpu_allocator;
  umpire::Allocator m_gpu_allocator;
  
  // Helper to fill array with test data
  void fillWithTestData(CopyHidingArray<int>& arr, int start_val = 1) {
    int* data = arr.data(ExecutionContext::CPU);
    for (size_t i = 0; i < arr.size(); ++i) {
      data[i] = start_val + i;
    }
  }
  
  // Helper to verify array contents
  void verifyArrayContents(const CopyHidingArray<int>& arr, int start_val = 1) {
    const int* data = arr.data(ExecutionContext::CPU);
    for (size_t i = 0; i < arr.size(); ++i) {
      EXPECT_EQ(data[i], start_val + i);
    }
  }
};

// Test default constructor
TEST_F(CopyHidingArrayTest, DefaultConstructor) {
  CopyHidingArray<int> arr;
  EXPECT_EQ(arr.size(), 0);
  EXPECT_EQ(arr.data(ExecutionContext::NONE), nullptr);
}

// Test constructor with allocators
TEST_F(CopyHidingArrayTest, AllocatorConstructor) {
  CopyHidingArray<int> arr(m_cpu_allocator, m_gpu_allocator);
  EXPECT_EQ(arr.size(), 0);
}

// Test size constructor
TEST_F(CopyHidingArrayTest, SizeConstructor) {
  const size_t size = 100;
  CopyHidingArray<int> arr(size);
  EXPECT_EQ(arr.size(), size);
}

// Test size and allocator constructor
TEST_F(CopyHidingArrayTest, SizeAndAllocatorConstructor) {
  const size_t size = 100;
  CopyHidingArray<int> arr(size, m_cpu_allocator, m_gpu_allocator);
  EXPECT_EQ(arr.size(), size);
}

// Test copy constructor
TEST_F(CopyHidingArrayTest, CopyConstructor) {
  const size_t size = 100;
  CopyHidingArray<int> arr1(size);
  fillWithTestData(arr1);
  
  CopyHidingArray<int> arr2(arr1);
  EXPECT_EQ(arr2.size(), size);
  verifyArrayContents(arr2);
}

// Test move constructor
TEST_F(CopyHidingArrayTest, MoveConstructor) {
  const size_t size = 100;
  CopyHidingArray<int> arr1(size);
  fillWithTestData(arr1);
  
  CopyHidingArray<int> arr2(std::move(arr1));
  EXPECT_EQ(arr2.size(), size);
  EXPECT_EQ(arr1.size(), 0);
  verifyArrayContents(arr2);
}

// Test copy assignment
TEST_F(CopyHidingArrayTest, CopyAssignment) {
  const size_t size = 100;
  CopyHidingArray<int> arr1(size);
  fillWithTestData(arr1);
  
  CopyHidingArray<int> arr2;
  arr2 = arr1;
  EXPECT_EQ(arr2.size(), size);
  verifyArrayContents(arr2);
}

// Test move assignment
TEST_F(CopyHidingArrayTest, MoveAssignment) {
  const size_t size = 100;
  CopyHidingArray<int> arr1(size);
  fillWithTestData(arr1);
  
  CopyHidingArray<int> arr2;
  arr2 = std::move(arr1);
  EXPECT_EQ(arr2.size(), size);
  EXPECT_EQ(arr1.size(), 0);
  verifyArrayContents(arr2);
}

// Test data access and coherence
TEST_F(CopyHidingArrayTest, DataCoherence) {
  const size_t size = 100;
  CopyHidingArray<int> arr(size);
  
  // Fill on CPU
  int* cpu_data = arr.data(ExecutionContext::CPU);
  for (size_t i = 0; i < size; ++i) {
    cpu_data[i] = i + 1;
  }
  
  // Access on GPU (this will cause a copy)
  int* gpu_data = arr.data(ExecutionContext::GPU);
  // Here we would run a GPU kernel, but for testing we'll just verify the copy happens
  
  // Modify on GPU (simulation for test)
  // In a real test, this would be done via a GPU kernel
  for (size_t i = 0; i < size; ++i) {
    gpu_data[i] *= 2;
  }
  
  // Access back on CPU (should trigger a copy back)
  cpu_data = arr.data(ExecutionContext::CPU);
  
  // Verify data was copied back correctly
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(cpu_data[i], (i + 1) * 2);
  }
}

// Test resize functionality
TEST_F(CopyHidingArrayTest, Resize) {
  const size_t initial_size = 50;
  const size_t new_size = 100;
  
  CopyHidingArray<int> arr(initial_size);
  fillWithTestData(arr);
  
  EXPECT_EQ(arr.size(), initial_size);
  
  arr.resize(new_size);
  EXPECT_EQ(arr.size(), new_size);
  
  // First initial_size elements should still have their values
  int* data = arr.data(ExecutionContext::CPU);
  for (size_t i = 0; i < initial_size; ++i) {
    EXPECT_EQ(data[i], i + 1);
  }
}

// Test const data access
TEST_F(CopyHidingArrayTest, ConstDataAccess) {
  const size_t size = 100;
  CopyHidingArray<int> arr(size);
  fillWithTestData(arr);
  
  const CopyHidingArray<int>& const_arr = arr;
  const int* const_data = const_arr.data(ExecutionContext::CPU);
  
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(const_data[i], i + 1);
  }
}

} // namespace expt
} // namespace chai