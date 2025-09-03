#include "gtest/gtest.h"
#include "chai/expt/UnifiedMemoryManager.hpp"

class UnifiedMemoryManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Get a basic allocator for testing
    m_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
    m_execution_context_manager = ExecutionContextManager::getInstance();
  }

  umpire::Allocator m_allocator;
  ExecutionContextManager& m_execution_context_manager;
};

TEST_F(UnifiedMemoryManagerTest, DefaultConstructor) {
  chai::expt::UnifiedMemoryManager manager;

  {
    EXPECT_EQ(manager.size(), 0);
    EXPECT_EQ(manager.data(), nullptr);
  }

  {
    chai::expt::ExecutionContextGuard executionContextGuard(ExecutionContext::NONE);
    EXPECT_EQ(manager.size(), 0);
    EXPECT_EQ(manager.data(), nullptr);
  }

  {
    chai::expt::ExecutionContextGuard executionContextGuard(ExecutionContext::HOST);
    EXPECT_EQ(manager.size(), 0);
    EXPECT_EQ(manager.data(), nullptr);
  }

#if defined(CHAI_ENABLE_DEVICE)
  {
    chai::expt::ExecutionContextGuard executionContextGuard(ExecutionContext::DEVICE);
    EXPECT_EQ(manager.size(), 0);
    EXPECT_EQ(manager.data(), nullptr);
  }
#endif
}

TEST_F(UnifiedMemoryManagerTest, AllocatorConstructor) {
  chai::expt::UnifiedMemoryManager manager(m_allocator);
  EXPECT_EQ(manager.size(), 0);
  EXPECT_EQ(manager.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, SizeAndAllocatorConstructor) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager manager(size, m_allocator);
  EXPECT_EQ(manager.size(), size);
  EXPECT_NE(manager.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, AllocatorIDConstructor) {
  chai::expt::UnifiedMemoryManager manager(0); // 0 typically corresponds to HOST
  EXPECT_EQ(manager.size(), 0);
  EXPECT_EQ(manager.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, SizeAndAllocatorIDConstructor) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager manager(size, 0); // 0 typically corresponds to HOST
  EXPECT_EQ(manager.size(), size);
  EXPECT_NE(manager.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, CopyConstructor) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager original(size, m_allocator);
  
  // Initialize data
  for (size_t i = 0; i < size; ++i) {
    original.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Copy
  chai::expt::UnifiedMemoryManager copy(original);
  
  // Verify
  EXPECT_EQ(copy.size(), original.size());
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(copy.get(chai::expt::ExecutionContext::HOST, i), 
              original.get(chai::expt::ExecutionContext::HOST, i));
  }
}

TEST_F(UnifiedMemoryManagerTest, MoveConstructor) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager original(size, m_allocator);
  
  // Initialize data
  for (size_t i = 0; i < size; ++i) {
    original.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Store data pointer for comparison
  int* originalData = original.data(chai::expt::ExecutionContext::HOST);
  
  // Move
  chai::expt::UnifiedMemoryManager moved(std::move(original));
  
  // Verify
  EXPECT_EQ(moved.size(), size);
  EXPECT_EQ(moved.data(chai::expt::ExecutionContext::HOST), originalData);
  EXPECT_EQ(original.size(), 0);
  EXPECT_EQ(original.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, CopyAssignment) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager original(size, m_allocator);
  
  // Initialize data
  for (size_t i = 0; i < size; ++i) {
    original.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Copy assignment
  chai::expt::UnifiedMemoryManager copy;
  copy = original;
  
  // Verify
  EXPECT_EQ(copy.size(), original.size());
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(copy.get(chai::expt::ExecutionContext::HOST, i), 
              original.get(chai::expt::ExecutionContext::HOST, i));
  }
}

TEST_F(UnifiedMemoryManagerTest, MoveAssignment) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager original(size, m_allocator);
  
  // Initialize data
  for (size_t i = 0; i < size; ++i) {
    original.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Store data pointer for comparison
  int* originalData = original.data(chai::expt::ExecutionContext::HOST);
  
  // Move assignment
  chai::expt::UnifiedMemoryManager moved;
  moved = std::move(original);
  
  // Verify
  EXPECT_EQ(moved.size(), size);
  EXPECT_EQ(moved.data(chai::expt::ExecutionContext::HOST), originalData);
  EXPECT_EQ(original.size(), 0);
  EXPECT_EQ(original.data(chai::expt::ExecutionContext::HOST), nullptr);
}

TEST_F(UnifiedMemoryManagerTest, DataAccess) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager manager(size, m_allocator);
  
  // Initialize and verify data access
  for (size_t i = 0; i < size; ++i) {
    manager.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Verify using get()
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(manager.get(chai::expt::ExecutionContext::HOST, i), static_cast(i));
  }
  
  // Verify using data()
  int* data = manager.data(chai::expt::ExecutionContext::HOST);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], static_cast(i));
  }
}

TEST_F(UnifiedMemoryManagerTest, ConstDataAccess) {
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager manager(size, m_allocator);
  
  // Initialize data
  for (size_t i = 0; i < size; ++i) {
    manager.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Create const reference and verify data access
  const chai::expt::UnifiedMemoryManager& constManager = manager;
  
  // Verify using get()
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(constManager.get(chai::expt::ExecutionContext::HOST, i), static_cast(i));
  }
  
  // Verify using data()
  const int* constData = constManager.data(chai::expt::ExecutionContext::HOST);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(constData[i], static_cast(i));
  }
}

TEST_F(UnifiedMemoryManagerTest, ExecutionContextSwitching) {
  // Note: This test assumes a system with both HOST and DEVICE execution contexts
  // For systems without a device (e.g., GPU), this test may need to be modified
  
  const size_t size = 10;
  chai::expt::UnifiedMemoryManager manager(size, m_allocator);
  
  // Initialize data on HOST
  for (size_t i = 0; i < size; ++i) {
    manager.get(chai::expt::ExecutionContext::HOST, i) = static_cast(i);
  }
  
  // Access data on DEVICE (should trigger synchronization)
  int* deviceData = manager.data(chai::expt::ExecutionContext::DEVICE);
  
  // Access data back on HOST (should trigger synchronization again)
  int* hostData = manager.data(chai::expt::ExecutionContext::HOST);
  
  // Verify data is still correct
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(hostData[i], static_cast(i));
  }
}