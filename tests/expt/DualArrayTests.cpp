#include "chai/expt/DualArray.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "gtest/gtest.h"

enum class ContextManagerState
{
  CONTEXT_NONE_SYNCHRONIZED_DEVICE,
  CONTEXT_HOST_SYNCHRONIZED_DEVICE,
  CONTEXT_DEVICE_SYNCHRONIZED_DEVICE,
  CONTEXT_NONE_UNSYNCHRONIZED_DEVICE,
  CONTEXT_HOST_UNSYNCHRONIZED_DEVICE,
  CONTEXT_DEVICE_UNSYNCHRONIZED_DEVICE
};

class DualArrayTest : public ::testing::Test
{
  protected:
    static chai::expt::ContextManager& GetContextManager()
    {
      static chai::expt::ContextManager& s_context_manager =
        chai::expt::ContextManager::getInstance();

      return s_context_manager;
    }

    static umpire::ResourceManager& GetResourceManager()
    {
      static umpire::ResourceManager& s_resource_manager =
        umpire::ResourceManager::getInstance();

      return s_resource_manager;
    }

    static umpire::Allocator& GetDefaultHostAllocator()
    {
      static umpire::Allocator s_default_host_allocator =
        GetResourceManager().getAllocator("HOST");

      return s_default_host_allocator;
    }

    static umpire::Allocator& GetDefaultDeviceAllocator()
    {
      static umpire::Allocator s_default_device_allocator =
        GetResourceManager().getAllocator("DEVICE");

      return s_default_device_allocator;
    }

    static umpire::Allocator& GetCustomHostAllocator()
    {
      static umpire::Allocator s_custom_host_allocator =
        GetResourceManager().makeAllocator<umpire::strategy::QuickPool>(
          "HOST_CUSTOM", GetDefaultHostAllocator());

      return s_custom_host_allocator;
    }

    static umpire::Allocator& GetCustomDeviceAllocator()
    {
      static umpire::Allocator s_custom_device_allocator =
        GetResourceManager().makeAllocator<umpire::strategy::QuickPool>(
          "DEVICE_CUSTOM", GetDefaultDeviceAllocator());

      return s_custom_device_allocator;
    }

    static const std::array<ContextManagerState, 6>& GetContextManagerStates()
    {
      static constexpr std::array<ContextManagerState, 6> s_context_manager_states{
        ContextManagerState::CONTEXT_NONE_SYNCHRONIZED_DEVICE,
        ContextManagerState::CONTEXT_HOST_SYNCHRONIZED_DEVICE,
        ContextManagerState::CONTEXT_DEVICE_SYNCHRONIZED_DEVICE,
        ContextManagerState::CONTEXT_NONE_UNSYNCHRONIZED_DEVICE,
        ContextManagerState::CONTEXT_HOST_UNSYNCHRONIZED_DEVICE,
        ContextManagerState::CONTEXT_DEVICE_UNSYNCHRONIZED_DEVICE
      };

      return s_context_manager_states;
    }

    void SetUp() override
    {
      GetContextManager().reset();
    }

    void TearDown() override
    {
      GetContextManager().reset();
    }

    void SetContextManagerState(ContextManagerState state)
    {
      chai::expt::ContextManager& context_manager = GetContextManager();
      context_manager.reset();

      if (state == ContextManagerState::CONTEXT_NONE_SYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::NONE);
      }
      else if (state == ContextManagerState::CONTEXT_HOST_SYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::HOST);
      }
      else if (state == ContextManagerState::CONTEXT_DEVICE_SYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::DEVICE);
        context_manager.synchronize(chai::expt::Context::DEVICE);
      }
      else if (state == ContextManagerState::CONTEXT_NONE_UNSYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::DEVICE);
        context_manager.setContext(chai::expt::Context::NONE);
      }
      else if (state == ContextManagerState::CONTEXT_HOST_UNSYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::DEVICE);
        context_manager.setContext(chai::expt::Context::HOST);
      }
      else if (state == ContextManagerState::CONTEXT_DEVICE_UNSYNCHRONIZED_DEVICE)
      {
        context_manager.setContext(chai::expt::Context::DEVICE);
      }
    }

    std::size_t m_size = 10;
};

TEST_F(DualArrayTest, DefaultConstructor)
{
  for (ContextManagerState context_manager_state : GetContextManagerStates())
  {
    SetContextManagerState(context_manager_state);
    chai::expt::DualArray<int> array;
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.modified(), chai::expt::Context::NONE);
    EXPECT_EQ(array.host_data(), nullptr);
    EXPECT_EQ(array.device_data(), nullptr);
    EXPECT_EQ(array.host_allocator().getId(), GetDefaultHostAllocator().getId());
    EXPECT_EQ(array.device_allocator().getId(), GetDefaultDeviceAllocator().getId());
  }
}

TEST_F(DualArrayTest, AllocatorConstructor)
{
  for (ContextManagerState context_manager_state : GetContextManagerStates())
  {
    SetContextManagerState(context_manager_state);
    chai::expt::DualArray<int> array(GetCustomHostAllocator(), GetCustomDeviceAllocator());
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.modified(), chai::expt::Context::NONE);
    EXPECT_EQ(array.host_data(), nullptr);
    EXPECT_EQ(array.device_data(), nullptr);
    EXPECT_EQ(array.host_allocator().getId(), GetCustomHostAllocator().getId());
    EXPECT_EQ(array.device_allocator().getId(), GetCustomDeviceAllocator().getId());
  }
}

TEST_F(DualArrayTest, SizeConstructor)
{
  for (ContextManagerState context_manager_state : GetContextManagerStates())
  {
    SetContextManagerState(context_manager_state);
    chai::expt::DualArray<int> array(m_size);
    EXPECT_EQ(array.size(), m_size);
    EXPECT_EQ(array.modified(), chai::expt::Context::NONE);

    if (context_manager_state == ContextManagerState::CONTEXT_NONE_SYNCHRONIZED_DEVICE ||
        context_manager_state == ContextManagerState::CONTEXT_NONE_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_manager_state == ContextManagerState::CONTEXT_HOST_SYNCHRONIZED_DEVICE ||
             context_manager_state == ContextManagerState::CONTEXT_HOST_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_NE(array.host_data(), nullptr);
      ASSERT_TRUE(GetResourceManager().hasAllocator(array.host_data()));
      EXPECT_EQ(GetResourceManager().getAllocator(array.host_data()).getId(), GetDefaultHostAllocator().getId());
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_manager_state == ContextManagerState::CONTEXT_DEVICE_SYNCHRONIZED_DEVICE ||
             context_manager_state == ContextManagerState::CONTEXT_DEVICE_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_NE(array.device_data(), nullptr);
      ASSERT_TRUE(GetResourceManager().hasAllocator(array.device_data()));
      EXPECT_EQ(GetResourceManager().getAllocator(array.device_data()).getId(), GetDefaultDeviceAllocator().getId());
    }
  }
}

TEST_F(DualArrayTest, SizeAndAllocatorConstructor)
{
  for (auto context_manager_state : GetContextManagerStates())
  {
    chai::expt::DualArray<int> array(m_size,
                                     GetCustomHostAllocator(),
                                     GetCustomDeviceAllocator());

    EXPECT_EQ(array.size(), m_size);
    EXPECT_EQ(array.modified(), chai::expt::Context::NONE);

    if (context_manager_state == ContextManagerState::CONTEXT_NONE_SYNCHRONIZED_DEVICE ||
        context_manager_state == ContextManagerState::CONTEXT_NONE_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_manager_state == ContextManagerState::CONTEXT_HOST_SYNCHRONIZED_DEVICE ||
             context_manager_state == ContextManagerState::CONTEXT_HOST_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_NE(array.host_data(), nullptr);
      ASSERT_TRUE(GetResourceManager().hasAllocator(array.host_data()));
      EXPECT_EQ(GetResourceManager().getAllocator(array.host_data()).getId(), GetCustomHostAllocator().getId());
      EXPECT_EQ(array.device_data(), nullptr);
    }
    else if (context_manager_state == ContextManagerState::CONTEXT_DEVICE_SYNCHRONIZED_DEVICE ||
             context_manager_state == ContextManagerState::CONTEXT_DEVICE_UNSYNCHRONIZED_DEVICE)
    {
      EXPECT_EQ(array.host_data(), nullptr);
      EXPECT_NE(array.device_data(), nullptr);
      ASSERT_TRUE(GetResourceManager().hasAllocator(array.device_data()));
      EXPECT_EQ(GetResourceManager().getAllocator(array.device_data()).getId(), GetCustomDeviceAllocator().getId());
    }
  }
}

TEST_F(DualArrayTest, CopyConstructor) {
  const size_t size = 5;
  chai::expt::DualArray<int> array1(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data in array1
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Copy construct array2
  chai::expt::DualArray<int> array2(array1);
  
  EXPECT_EQ(array2.size(), size);
  
  // Verify data was copied
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, MoveConstructor) {
  const size_t size = 5;
  chai::expt::DualArray<int> array1(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data in array1
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Move construct array2
  chai::expt::DualArray<int> array2(std::move(array1));
  
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
  chai::expt::DualArray<int> array1(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data in array1
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Copy assign to array2
  chai::expt::DualArray<int> array2;
  array2 = array1;
  
  EXPECT_EQ(array2.size(), size);
  
  // Verify data was copied
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array2.get(i), static_cast<int>(i));
  }
}

TEST_F(DualArrayTest, MoveAssignment) {
  const size_t size = 5;
  chai::expt::DualArray<int> array1(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data in array1
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array1.set(i, static_cast<int>(i));
  }
  
  // Move assign to array2
  chai::expt::DualArray<int> array2;
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
  chai::expt::DualArray<int> array(5, GetCustomHostAllocator(), GetCustomDeviceAllocator());
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
  chai::expt::DualArray<int> array(initial_size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
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
  chai::expt::DualArray<int> array(5, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  array.free();
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.host_data(), nullptr);
  EXPECT_EQ(array.device_data(), nullptr);
  EXPECT_EQ(array.modified(), chai::expt::Context::NONE);
}

TEST_F(DualArrayTest, DataAndModified) {
  const size_t size = 5;
  chai::expt::DualArray<int> array(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Test host context
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  int* host_ptr = array.data();
  EXPECT_NE(host_ptr, nullptr);
  EXPECT_EQ(array.modified(), chai::expt::Context::HOST);
  
  // Test device context
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::DEVICE);
  int* device_ptr = array.data();
  EXPECT_NE(device_ptr, nullptr);
  EXPECT_EQ(array.modified(), chai::expt::Context::DEVICE);
}

TEST_F(DualArrayTest, ConstData) {
  const size_t size = 5;
  chai::expt::DualArray<int> array(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set some data
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array.set(i, static_cast<int>(i));
  }
  
  // Create a const reference
  const chai::expt::DualArray<int>& const_array = array;
  
  // Test host context
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  const int* host_ptr = const_array.data();
  EXPECT_NE(host_ptr, nullptr);
  
  // Test device context
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::DEVICE);
  const int* device_ptr = const_array.data();
  EXPECT_NE(device_ptr, nullptr);
}

TEST_F(DualArrayTest, GetAndSet) {
  const size_t size = 5;
  chai::expt::DualArray<int> array(size, GetCustomHostAllocator(), GetCustomDeviceAllocator());
  
  // Set values in host context
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    array.set(i, static_cast<int>(i * 10));
  }
  
  // Get values in host context
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i * 10));
  }
  
  // Switch to device context and test data sync
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::DEVICE);
  int* device_ptr = array.data();
  
  // Switch back to host and verify data still accessible
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(array.get(i), static_cast<int>(i * 10));
  }
}