#include "gtest/gtest.h"
#include "chai/expt/DualArray.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai::expt {

class DualArrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    m_host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
    m_device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
  }

  umpire::Allocator m_host_allocator;
  umpire::Allocator m_device_allocator;
};

TEST_F(DualArrayTest, DefaultConstructor) {
  DualArray<int> array;
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.modified(), Context::NONE);
  EXPECT_EQ(array.host_data(), nullptr);
  EXPECT_EQ(array.device_data(), nullptr);
}

TEST_F(DualArrayTest, AllocatorConstructor) {
  DualArray<int> array(m_host_allocator, m_device_allocator);
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.modified(), Context::NONE);
  EXPECT_EQ(array.host_data(), nullptr);
  EXPECT_EQ(array.device_data(), nullptr);
}

TEST_F(DualArrayTest, SizeAndAllocatorConstructor) {
  const size_t size = 10;
  DualArray<int> array(size, m_host_allocator, m_device_allocator);
  EXPECT_EQ(array.size(), size);
  EXPECT_EQ(array.modified(), Context::NONE);
}

TEST_F(DualArrayTest, CopyConstructor) {
  const size_t size = 5;
  DualArray<int> array1(size, m_host_allocator, m_device_allocator);
  
  // Set some data in array1
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Copy construct array2
  DualArray<int> array2(array1);
  
  EXPECT_EQ(array2.size(), size);
  
  // Verify data was copied
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, MoveConstructor) {
  const size_t size = 5;
  DualArray<int> array1(size, m_host_allocator, m_device_allocator);
  
  // Set some data in array1
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Move construct array2
  DualArray<int> array2(std::move(array1));
  
  EXPECT_EQ(array2.size(), size);
  EXPECT_EQ(array1.size(), 0);  // array1 should be empty after move
  EXPECT_EQ(array1.host_data(), nullptr);
  EXPECT_EQ(array1.device_data(), nullptr);
  
  // Verify data was moved
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, CopyAssignment) {
  const size_t size = 5;
  DualArray<int> array1(size, m_host_allocator, m_device_allocator);
  
  // Set some data in array1
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Copy assign to array2
  DualArray<int> array2;
  array2 = array1;
  
  EXPECT_EQ(array2.size(), size);
  
  // Verify data was copied
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, MoveAssignment) {
  const size_t size = 5;
  DualArray<int> array1(size, m_host_allocator, m_device_allocator);
  
  // Set some data in array1
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Move assign to array2
  DualArray<int> array2;
  array2 = std::move(array1);
  
  EXPECT_EQ(array2.size(), size);
  EXPECT_EQ(array1.size(), 0);  // array1 should be empty after move
  EXPECT_EQ(array1.host_data(), nullptr);
  EXPECT_EQ(array1.device_data(), nullptr);
  
  // Verify data was moved
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, Resize) {
  DualArray<int> array(5, m_host_allocator, m_device_allocator);
  EXPECT_EQ(array.size(), 5);
  
  // Resize larger
  array.resize(10);
  EXPECT_EQ(array.size(), 10);
  
  // Resize smaller
  array.resize(3);
  EXPECT_EQ(array.size(), 3);
  
  // Resize to zero
  array.resize(0);
  EXPECT_EQ(array.size(), 0);
}

TEST_F(DualArrayTest, ResizeWithData) {
  const size_t initial_size = 5;
  DualArray<int> array(initial_size, m_host_allocator, m_device_allocator);
  
  // Set some data
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < initial_size; ++i) {
    array.set(i, static_cast<int>(i));
  }
  
  // Resize larger
  const size_t new_size = 8;
  array.resize(new_size);
  
  // Verify original data is preserved
  for (size_t i = 0; i < initial_size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i));
  }
  
  // Resize smaller
  const size_t smaller_size = 3;
  array.resize(smaller_size);
  
  // Verify remaining data is preserved
  for (size_t i = 0; i < smaller_size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, Free) {
  DualArray<int> array(5, m_host_allocator, m_device_allocator);
  
  array.free();
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.host_data(), nullptr);
  EXPECT_EQ(array.device_data(), nullptr);
  EXPECT_EQ(array.modified(), Context::NONE);
}

TEST_F(DualArrayTest, DataAndModified) {
  const size_t size = 5;
  DualArray<int> array(size, m_host_allocator, m_device_allocator);
  
  // Test host context
  ContextManager::getInstance().setContext(Context::HOST);
  int* host_ptr = array.data();
  EXPECT_NE(host_ptr, nullptr);
  EXPECT_EQ(array.modified(), Context::HOST);
  
  // Test device context
  ContextManager::getInstance().setContext(Context::DEVICE);
  int* device_ptr = array.data();
  EXPECT_NE(device_ptr, nullptr);
  EXPECT_EQ(array.modified(), Context::DEVICE);
}

TEST_F(DualArrayTest, ConstData) {
  const size_t size = 5;
  DualArray<int> array(size, m_host_allocator, m_device_allocator);
  
  // Set some data
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array.set(i, static_cast<int>(i));
  }
  
  // Create a const reference
  const DualArray<int>& const_array = array;
  
  // Test host context
  ContextManager::getInstance().setContext(Context::HOST);
  const int* host_ptr = const_array.data();
  EXPECT_NE(host_ptr, nullptr);
  
  // Test device context
  ContextManager::getInstance().setContext(Context::DEVICE);
  const int* device_ptr = const_array.data();
  EXPECT_NE(device_ptr, nullptr);
}

TEST_F(DualArrayTest, GetAndSet) {
  const size_t size = 5;
  DualArray<int> array(size, m_host_allocator, m_device_allocator);
  
  // Set values in host context
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array.set(i, static_cast<int>(i * 10));
  }
  
  // Get values in host context
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i * 10));
  }
  
  // Switch to device context and test data sync
  ContextManager::getInstance().setContext(Context::DEVICE);
  int* device_ptr = array.data();
  
  // Switch back to host and verify data still accessible
  ContextManager::getInstance().setContext(Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i * 10));
  }
}

} // namespace chai::expt