#include "chai/expt/DualArray.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/ResourceManager.hpp"
#include "gtest/gtest.h"

enum class ContextState
{
  CONTEXT_NONE_DEVICE_SYNCHRONIZED,
  CONTEXT_HOST_DEVICE_SYNCHRONIZED,
  CONTEXT_DEVICE_DEVICE_SYNCHRONIZED,
  CONTEXT_NONE_DEVICE_UNSYNCHRONIZED,
  CONTEXT_HOST_DEVICE_UNSYNCHRONIZED,
  CONTEXT_DEVICE_DEVICE_UNSYNCHRONIZED
};

class ContextIterator
{
  public:

  private:
    chai::expt::ContextManager& m_context_manager{chai::expt::ContextManager::getInstance()};
    ContextState m_context_state = CONTEXT_NONE_DEVICE_SYNCHRONIZED;

};

class DualArrayTest : public ::testing::Test
{
  protected:
    static void SetUpTestSuite()
    {
      auto& rm = umpire::ResourceManager::getInstance();

      m_default_host_allocator = rm.getAllocator("HOST");
      m_default_device_allocator = rm.getAllocator("DEVICE");

      m_custom_host_allocator =
        rm.makeAllocator<umpire::strategy::QuickPool>(
          "HOST_CUSTOM", m_default_host_allocator);

      m_custom_device_allocator =
        rm.makeAllocator<umpire::strategy::QuickPool>(
          "DEVICE_CUSTOM", m_default_device_allocator);
    }

    void SetUp() override {
      m_context_manager.reset();
    }

    void SetContext(chai::expt::Context context, bool device_synchronized)
    {
      m_context_manager.setContext(context);
      m_context_manager.setSynchronized(chai::expt::Context::DEVICE, device_synchronized);
    }

    umpire::Allocator m_default_host_allocator;
    umpire::Allocator m_default_device_allocator;
    
    umpire::Allocator m_custom_host_allocator;
    umpire::Allocator m_custom_device_allocator;

    chai::expt::ContextManager& m_context_manager{chai::expt::ContextManager::getInstance()};

    m_size = 10;

    std::array<std::pair<chai::expt::Context, bool>, 6> m_context_states = {{
      {chai::expt::Context::NONE,   false},
      {chai::expt::Context::HOST,   false},
      {chai::expt::Context::DEVICE, false}
      {chai::expt::Context::NONE,   true},
      {chai::expt::Context::HOST,   true},
      {chai::expt::Context::DEVICE, true}
    }};
};

TEST_F(DualArrayTest, DefaultConstructor)
{
  for (auto context_state : m_context_states)
  {
    chai::expt::DualArray<int> array;
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.modified(), Context::NONE);
    EXPECT_EQ(array.host_data(), nullptr);
    EXPECT_EQ(array.device_data(), nullptr);
    EXPECT_EQ(array.host_allocator.getId(), m_default_host_allocator.getId());
    EXPECT_EQ(array.device_allocator.getId(), m_default_device_allocator.getId());
  }
}

TEST_F(DualArrayTest, AllocatorConstructor)
{
  for (auto context_state : m_context_states)
  {
    chai::expt::DualArray<int> array(m_custom_host_allocator, m_custom_device_allocator);
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.modified(), Context::NONE);
    EXPECT_EQ(array.host_data(), nullptr);
    EXPECT_EQ(array.device_data(), nullptr);
    EXPECT_EQ(array.host_allocator.getId(), m_custom_host_allocator.getId());
    EXPECT_EQ(array.device_allocator.getId(), m_custom_device_allocator.getId());
  }
}

TEST_F(DualArrayTest, SizeConstructor)
{
  for (auto context_state : m_context_states)
  {
    chai::expt::DualArray<int> array(m_size);
    EXPECT_EQ(array.size(), m_size);
    EXPECT_EQ(array.modified(), Context::NONE);

    if (context_state == ContextState::CONTEXT_NONE_DEVICE_SYNCHRONIZED ||
        context_state == ContextState::CONTEXT_NONE_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_state == ContextState::CONTEXT_HOST_DEVICE_SYNCHRONIZED ||
             context_state == ContextState::CONTEXT_HOST_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_NE(array.host_data(), nullptr);
      ASSERT_TRUE(m_resource_manager.hasAllocator(array.host_data()));
      EXPECT_EQ(m_resource_manager.getAllocator(array.host_data().getID(), m_default_host_allocator.getID()));
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_state == ContextState::CONTEXT_DEVICE_DEVICE_SYNCHRONIZED ||
             context_state == ContextState::CONTEXT_DEVICE_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_NE(array.device_data(), nullptr);
      ASSERT_TRUE(m_resource_manager.hasAllocator(array.device_data()));
      EXPECT_EQ(m_resource_manager.getAllocator(array.device_data().getID(), m_default_device_allocator.getID()));
    }
  }
}

TEST_F(DualArrayTest, SizeAndAllocatorConstructor)
{
  for (auto context_state : m_context_states)
  {
    chai::expt::DualArray<int> array(m_size,
                                     m_custom_host_allocator,
                                     m_custom_device_allocator);

    EXPECT_EQ(array.size(), m_size);
    EXPECT_EQ(array.modified(), Context::NONE);

    if (context_state == ContextState::CONTEXT_NONE_DEVICE_SYNCHRONIZED ||
        context_state == ContextState::CONTEXT_NONE_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_state == ContextState::CONTEXT_HOST_DEVICE_SYNCHRONIZED ||
             context_state == ContextState::CONTEXT_HOST_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_NE(array.host_data(), nullptr);
      ASSERT_TRUE(m_resource_manager.hasAllocator(array.host_data()));
      EXPECT_EQ(m_resource_manager.getAllocator(array.host_data().getID(), m_default_host_allocator.getID()));
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_state == ContextState::CONTEXT_DEVICE_DEVICE_SYNCHRONIZED ||
             context_state == ContextState::CONTEXT_DEVICE_DEVICE_UNSYNCHRONIZED)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_NE(array.device_data(), nullptr);
      ASSERT_TRUE(m_resource_manager.hasAllocator(array.device_data()));
      EXPECT_EQ(m_resource_manager.getAllocator(array.device_data().getID(), m_default_device_allocator.getID()));
    }
  }
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