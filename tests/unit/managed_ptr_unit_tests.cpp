//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X_##Y();    \
  TEST(X, Y) { gpu_test_##X_##Y(); } \
  static void gpu_test_##X_##Y()

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/NamingShim.hpp"

#include "../src/util/forall.hpp"

// Standard library headers
#include <cstdlib>
#include <new>

class Simple {
   public:
      CHAI_HOST_DEVICE Simple() : m_value(-1) {}
      CHAI_HOST_DEVICE Simple(int value) : m_value(value) {}
      CHAI_HOST_DEVICE ~Simple() {}

      CHAI_HOST_DEVICE Simple(Simple const & other) : m_value(other.m_value) {}

      CHAI_HOST_DEVICE Simple& operator=(Simple const & other) {
         m_value = other.m_value;
         return *this;
      }

      CHAI_HOST_DEVICE Simple(Simple&& other) : m_value(other.m_value) {
         other.m_value = -1;
      }

      CHAI_HOST_DEVICE Simple& operator=(Simple&& other) {
         m_value = other.m_value;
         other.m_value = -1;
         return *this;
      }

      CHAI_HOST_DEVICE int getValue() { return m_value; }

   private:
      int m_value;
};

class TestBase {
   public:
      CHAI_HOST_DEVICE TestBase() {}

      CHAI_HOST_DEVICE virtual ~TestBase() {}

      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived() : TestBase(), m_value(0) {}
      CHAI_HOST_DEVICE TestDerived(const int value) : TestBase(), m_value(value) {}

      CHAI_HOST_DEVICE virtual ~TestDerived() {}

      CHAI_HOST_DEVICE virtual int getValue() const { return m_value; }

   private:
      int m_value;
};

class AllocatorTracked {
   public:
      CHAI_HOST_DEVICE AllocatorTracked(int value, int* destructionCount) :
         m_value(value),
         m_destruction_count(destructionCount)
      {
      }

      CHAI_HOST_DEVICE ~AllocatorTracked() {
#if !defined(CHAI_DEVICE_COMPILE)
         if (m_destruction_count) {
            ++(*m_destruction_count);
         }
#endif
      }

      CHAI_HOST_DEVICE int getValue() const { return m_value; }

   private:
      int m_value;
      int* m_destruction_count;
};

class AllocatorBase1 {
   public:
      CHAI_HOST_DEVICE virtual ~AllocatorBase1() {}
      CHAI_HOST_DEVICE virtual int getBase1Value() const = 0;
};

class AllocatorBase2 {
   public:
      CHAI_HOST_DEVICE virtual ~AllocatorBase2() {}
      CHAI_HOST_DEVICE virtual int getBase2Value() const = 0;
};

class AllocatorDerived : public AllocatorBase1, public AllocatorBase2 {
   public:
      CHAI_HOST_DEVICE AllocatorDerived(int value, int* destructionCount) :
         m_value(value),
         m_destruction_count(destructionCount)
      {
      }

      CHAI_HOST_DEVICE ~AllocatorDerived() override {
#if !defined(CHAI_DEVICE_COMPILE)
         if (m_destruction_count) {
            ++(*m_destruction_count);
         }
#endif
      }

      CHAI_HOST_DEVICE int getBase1Value() const override { return m_value; }
      CHAI_HOST_DEVICE int getBase2Value() const override { return m_value; }

   private:
      int m_value;
      int* m_destruction_count;
};

class DeviceAllocatorBase {
   public:
      CHAI_HOST_DEVICE virtual ~DeviceAllocatorBase() {}
      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class DeviceAllocatorDerived : public DeviceAllocatorBase {
   public:
      CHAI_HOST_DEVICE DeviceAllocatorDerived(int value, int* destructionCount) :
         m_value(value),
         m_destruction_count(destructionCount)
      {
      }

      CHAI_HOST_DEVICE ~DeviceAllocatorDerived() override {
         if (m_destruction_count) {
            ++(*m_destruction_count);
         }
      }

      CHAI_HOST_DEVICE int getValue() const override { return m_value; }

   private:
      int m_value;
      int* m_destruction_count;
};

class ClassSpecificAllocation {
   public:
      static void* operator new(std::size_t size) {
         ++s_new_calls;
         return ::operator new(size);
      }

      static void operator delete(void* pointer) {
         ++s_delete_calls;
         ::operator delete(pointer);
      }

      CHAI_HOST_DEVICE explicit ClassSpecificAllocation(int value) :
         m_value(value)
      {
      }

      CHAI_HOST_DEVICE ~ClassSpecificAllocation() {}

      CHAI_HOST_DEVICE int getValue() const { return m_value; }

      static int s_new_calls;
      static int s_delete_calls;

   private:
      int m_value;
};

int ClassSpecificAllocation::s_new_calls = 0;
int ClassSpecificAllocation::s_delete_calls = 0;

umpire::Allocator getManagedPtrTestHostAllocator()
{
  auto& resourceManager = umpire::ResourceManager::getInstance();
  static auto allocator =
      resourceManager.makeAllocator<umpire::strategy::NamingShim>(
          "chai_managed_ptr_host_allocator",
          resourceManager.getAllocator("HOST"));
  return allocator;
}

TEST(managed_ptr, default_constructor)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_FALSE(derived);
  EXPECT_TRUE(derived == nullptr);
  EXPECT_TRUE(nullptr == derived);
  EXPECT_FALSE(derived != nullptr);
  EXPECT_FALSE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(derived != otherDerived);
  EXPECT_FALSE(otherDerived != derived);

  derived.free();
  otherDerived.free();
}

TEST(managed_ptr, nullptr_constructor)
{
  chai::managed_ptr<TestDerived> derived = nullptr;
  chai::managed_ptr<TestDerived> otherDerived = nullptr;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_FALSE(derived);
  EXPECT_TRUE(derived == nullptr);
  EXPECT_TRUE(nullptr == derived);
  EXPECT_FALSE(derived != nullptr);
  EXPECT_FALSE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(derived != otherDerived);
  EXPECT_FALSE(otherDerived != derived);

  otherDerived.free();
  derived.free();
}

TEST(managed_ptr, cpu_pointer_constructor)
{
  TestDerived* cpuPointer = new TestDerived(3);
  chai::managed_ptr<TestDerived> derived({chai::CPU}, {cpuPointer});

  EXPECT_EQ(derived->getValue(), 3);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);

  derived.free();
}

TEST(managed_ptr, make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);

  derived.free();
}

TEST(managed_ptr, custom_deleter_constructor)
{
  auto allocator = getManagedPtrTestHostAllocator();
  const auto allocationsBefore = allocator.getAllocationCount();
  int destructionCount = 0;
  int deleterCount = 0;

  void* allocation = allocator.allocate(sizeof(AllocatorTracked));
  auto* object = ::new (allocation) AllocatorTracked(13, &destructionCount);
  chai::managed_ptr<AllocatorTracked> pointer(
      {chai::CPU}, {object},
      {[allocator, &deleterCount](AllocatorTracked* managedObject) mutable {
        managedObject->~AllocatorTracked();
        allocator.deallocate(managedObject);
        ++deleterCount;
      }});

  EXPECT_EQ(pointer->getValue(), 13);
  pointer.free();

  EXPECT_EQ(deleterCount, 1);
  EXPECT_EQ(destructionCount, 1);
  EXPECT_EQ(allocator.getAllocationCount(), allocationsBefore);
}

TEST(managed_ptr, allocate_managed_uses_cpu_allocator)
{
  auto allocator = getManagedPtrTestHostAllocator();
  const auto allocationsBefore = allocator.getAllocationCount();
  int destructionCount = 0;
  int cpuFreeCallbacks = 0;

  auto pointer = chai::allocate_managed<AllocatorTracked>(
      {chai::CPU}, {allocator}, 17, &destructionCount);

  EXPECT_EQ(pointer->getValue(), 17);
  EXPECT_EQ(allocator.getAllocationCount(), allocationsBefore + 1);

  pointer.set_callback(
      [&cpuFreeCallbacks](chai::Action action,
                          chai::ExecutionSpace space,
                          void*) {
        if (action == chai::ACTION_FREE && space == chai::CPU) {
          ++cpuFreeCallbacks;
        }
        return false;
      });
  pointer.free();

  EXPECT_EQ(cpuFreeCallbacks, 1);
  EXPECT_EQ(destructionCount, 1);
  EXPECT_EQ(allocator.getAllocationCount(), allocationsBefore);
}

TEST(managed_ptr, allocate_managed_callback_can_handle_cleanup)
{
  auto allocator = getManagedPtrTestHostAllocator();
  const auto allocationsBefore = allocator.getAllocationCount();
  int destructionCount = 0;
  int handledFreeCallbacks = 0;

  auto pointer = chai::allocate_managed<AllocatorTracked>(
      {chai::CPU}, {allocator}, 23, &destructionCount);

  pointer.set_callback(
      [allocator, &handledFreeCallbacks](chai::Action action,
                                         chai::ExecutionSpace space,
                                         void* rawPointer) mutable {
        if (action == chai::ACTION_FREE && space == chai::CPU) {
          auto* object = static_cast<AllocatorTracked*>(rawPointer);
          object->~AllocatorTracked();
          allocator.deallocate(object);
          ++handledFreeCallbacks;
          return true;
        }
        return false;
      });
  pointer.free();

  EXPECT_EQ(handledFreeCallbacks, 1);
  EXPECT_EQ(destructionCount, 1);
  EXPECT_EQ(allocator.getAllocationCount(), allocationsBefore);
}

TEST(managed_ptr, allocate_managed_bypasses_class_specific_new_and_delete)
{
  auto allocator = getManagedPtrTestHostAllocator();
  ClassSpecificAllocation::s_new_calls = 0;
  ClassSpecificAllocation::s_delete_calls = 0;

  auto pointer = chai::allocate_managed<ClassSpecificAllocation>(
      {chai::CPU}, {allocator}, 27);

  EXPECT_EQ(pointer->getValue(), 27);
  pointer.free();

  EXPECT_EQ(ClassSpecificAllocation::s_new_calls, 0);
  EXPECT_EQ(ClassSpecificAllocation::s_delete_calls, 0);
}

TEST(managed_ptr, allocate_managed_frees_original_pointer_after_conversion_and_cast)
{
  auto allocator = getManagedPtrTestHostAllocator();
  const auto allocationsBefore = allocator.getAllocationCount();
  int destructionCount = 0;

  auto derived = chai::allocate_managed<AllocatorDerived>(
      {chai::CPU}, {allocator}, 29, &destructionCount);
  chai::managed_ptr<AllocatorBase1> base1 = derived;
  auto base2 = chai::static_pointer_cast<AllocatorBase2>(derived);

  EXPECT_EQ(base1->getBase1Value(), 29);
  EXPECT_EQ(base2->getBase2Value(), 29);
  EXPECT_NE(static_cast<void*>(derived.get()),
            static_cast<void*>(base2.get()));

  base2.free();

  EXPECT_EQ(destructionCount, 1);
  EXPECT_EQ(allocator.getAllocationCount(), allocationsBefore);
}

#if !defined(CHAI_DISABLE_RM) || defined(CHAI_THIN_GPU_ALLOCATE)
TEST(managed_ptr, make_managed_uses_configured_cpu_allocator)
{
  auto* arrayManager = chai::ArrayManager::getInstance();
  auto originalAllocator = arrayManager->getAllocator(chai::CPU);
  auto configuredAllocator = getManagedPtrTestHostAllocator();
  const auto allocationsBefore = configuredAllocator.getAllocationCount();

  arrayManager->setAllocator(chai::CPU, configuredAllocator);
  auto pointer = chai::make_managed<AllocatorTracked>(31, nullptr);

  EXPECT_EQ(pointer->getValue(), 31);
  EXPECT_EQ(configuredAllocator.getAllocationCount(), allocationsBefore + 1);

  pointer.free();
  arrayManager->setAllocator(chai::CPU, originalAllocator);

  EXPECT_EQ(configuredAllocator.getAllocationCount(), allocationsBefore);
}
#endif

TEST(managed_ptr, copy_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived(derived);

  EXPECT_EQ(derived->getValue(), expectedValue);
  EXPECT_EQ(otherDerived->getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_FALSE(derived != otherDerived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_TRUE(otherDerived);
  EXPECT_FALSE(otherDerived == nullptr);
  EXPECT_FALSE(nullptr == otherDerived);
  EXPECT_TRUE(otherDerived != nullptr);
  EXPECT_TRUE(nullptr != otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(otherDerived != derived);

  derived.free();
}

TEST(managed_ptr, converting_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestBase> base = derived;

  EXPECT_EQ(derived->getValue(), expectedValue);
  EXPECT_EQ(base->getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == base);
  EXPECT_FALSE(derived != base);

  EXPECT_NE(base.get(), nullptr);
  EXPECT_TRUE(base);
  EXPECT_FALSE(base == nullptr);
  EXPECT_FALSE(nullptr == base);
  EXPECT_TRUE(base != nullptr);
  EXPECT_TRUE(nullptr != base);
  EXPECT_TRUE(base == derived);
  EXPECT_FALSE(base != derived);

  base.free();
}

TEST(managed_ptr, copy_assignment_operator)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  EXPECT_EQ(derived->getValue(), expectedValue);
  EXPECT_EQ(otherDerived->getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_FALSE(derived != otherDerived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_TRUE(otherDerived);
  EXPECT_FALSE(otherDerived == nullptr);
  EXPECT_FALSE(nullptr == otherDerived);
  EXPECT_TRUE(otherDerived != nullptr);
  EXPECT_TRUE(nullptr != otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(otherDerived != derived);

  otherDerived.free();
}

TEST(managed_ptr, copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived(derived);

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);

  derived.free();
}

TEST(managed_ptr, copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);

  otherDerived.free();
}

TEST(managed_ptr, conversion_copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived(derived);

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);

  otherDerived.free();
}

TEST(managed_ptr, conversion_copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived;
  otherDerived = derived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);

  derived.free();
}

TEST(managed_ptr, copy_assignment_operator_from_host_ptr_constructed)
{
  const int expectedValue1 = rand();
  const int expectedValue2 = rand();

  chai::managed_ptr<TestDerived> derived = chai::make_managed<TestDerived>(expectedValue1);
  chai::managed_ptr<TestDerived> otherDerived = chai::make_managed<TestDerived>(expectedValue2);
  chai::managed_ptr<TestDerived> thirdDerived(otherDerived);

  thirdDerived = derived;

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), true);
  EXPECT_NE(derived, nullptr);
  EXPECT_NE(nullptr, derived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), true);
  EXPECT_NE(otherDerived, nullptr);
  EXPECT_NE(nullptr, otherDerived);

  EXPECT_NE(thirdDerived.get(), nullptr);
  EXPECT_EQ(bool(thirdDerived), true);
  EXPECT_NE(thirdDerived, nullptr);
  EXPECT_NE(nullptr, thirdDerived);

  derived.free();
  otherDerived.free();
}

TEST(managed_ptr, conversion_copy_assignment_operator_from_host_ptr_constructed)
{
  const int expectedValue1 = rand();
  const int expectedValue2 = rand();

  chai::managed_ptr<TestDerived> derived = chai::make_managed<TestDerived>(expectedValue1);
  chai::managed_ptr<TestDerived> otherDerived = chai::make_managed<TestDerived>(expectedValue2);
  chai::managed_ptr<TestBase> thirdDerived(otherDerived);

  thirdDerived = derived;

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(bool(derived), true);
  EXPECT_NE(derived, nullptr);
  EXPECT_NE(nullptr, derived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(bool(otherDerived), true);
  EXPECT_NE(otherDerived, nullptr);
  EXPECT_NE(nullptr, otherDerived);

  EXPECT_NE(thirdDerived.get(), nullptr);
  EXPECT_EQ(bool(thirdDerived), true);
  EXPECT_NE(thirdDerived, nullptr);
  EXPECT_NE(nullptr, thirdDerived);

  otherDerived.free();
  thirdDerived.free();
}

TEST(managed_ptr, static_pointer_cast)
{
  TestDerived* cpuPointer = new TestDerived(3);
  chai::managed_ptr<TestDerived> derived({chai::CPU}, {cpuPointer});

  auto base = chai::static_pointer_cast<TestBase>(derived);

  EXPECT_EQ(base->getValue(), 3);

  EXPECT_NE(base.get(), nullptr);
  EXPECT_TRUE(base);
  EXPECT_FALSE(base == nullptr);
  EXPECT_FALSE(nullptr == base);
  EXPECT_TRUE(base != nullptr);
  EXPECT_TRUE(nullptr != base);

  base.free();
}

TEST(managed_ptr, dynamic_pointer_cast)
{
  TestDerived* cpuPointer = new TestDerived(3);
  chai::managed_ptr<TestBase> base({chai::CPU}, {cpuPointer});

  auto derived = chai::dynamic_pointer_cast<TestDerived>(base);

  EXPECT_EQ(derived->getValue(), 3);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);

  derived.free();
}

TEST(managed_ptr, const_pointer_cast)
{
  TestDerived* cpuPointer = new TestDerived(3);
  chai::managed_ptr<const TestBase> base({chai::CPU}, {cpuPointer});

  auto nonConstBase = chai::const_pointer_cast<TestBase>(base);

  EXPECT_EQ(nonConstBase->getValue(), 3);

  EXPECT_NE(nonConstBase.get(), nullptr);
  EXPECT_TRUE(nonConstBase);
  EXPECT_FALSE(nonConstBase == nullptr);
  EXPECT_FALSE(nullptr == nonConstBase);
  EXPECT_TRUE(nonConstBase != nullptr);
  EXPECT_TRUE(nullptr != nonConstBase);

  base.free();
}

TEST(managed_ptr, reinterpret_pointer_cast)
{
  TestDerived* cpuPointer = new TestDerived(3);
  chai::managed_ptr<TestBase> base({chai::CPU}, {cpuPointer});

  auto derived = chai::reinterpret_pointer_cast<TestDerived>(base);

  EXPECT_EQ(derived->getValue(), 3);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);

  derived.free();
}

#ifdef CHAI_GPUCC

GPU_TEST(managed_ptr, gpu_default_constructor)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;

  chai::ManagedArray<TestDerived*> array(1, chai::GPU);
  chai::ManagedArray<bool> array2(9, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived.get();
    array2[0] = (bool) derived;
    array2[1] = derived == nullptr;
    array2[2] = nullptr == derived;
    array2[3] = derived != nullptr;
    array2[4] = nullptr != derived;
    array2[5] = derived == otherDerived;
    array2[6] = otherDerived == derived;
    array2[7] = derived != otherDerived;
    array2[8] = otherDerived != derived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);

  EXPECT_EQ(array[0], nullptr);
  EXPECT_FALSE(array2[0]);
  EXPECT_TRUE(array2[1]);
  EXPECT_TRUE(array2[2]);
  EXPECT_FALSE(array2[3]);
  EXPECT_FALSE(array2[4]);
  EXPECT_TRUE(array2[5]);
  EXPECT_TRUE(array2[6]);
  EXPECT_FALSE(array2[7]);
  EXPECT_FALSE(array2[8]);

  array2.free();
  array.free();

  derived.free();
  otherDerived.free();
}

GPU_TEST(managed_ptr, gpu_nullptr_constructor)
{
  chai::managed_ptr<TestDerived> derived = nullptr;
  chai::managed_ptr<TestDerived> otherDerived = nullptr;

  chai::ManagedArray<TestDerived*> array(1, chai::GPU);
  chai::ManagedArray<bool> array2(9, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived.get();
    array2[0] = (bool) derived;
    array2[1] = derived == nullptr;
    array2[2] = nullptr == derived;
    array2[3] = derived != nullptr;
    array2[4] = nullptr != derived;
    array2[5] = derived == otherDerived;
    array2[6] = otherDerived == derived;
    array2[7] = derived != otherDerived;
    array2[8] = otherDerived != derived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);

  EXPECT_EQ(array[0], nullptr);
  EXPECT_FALSE(array2[0]);
  EXPECT_TRUE(array2[1]);
  EXPECT_TRUE(array2[2]);
  EXPECT_FALSE(array2[3]);
  EXPECT_FALSE(array2[4]);
  EXPECT_TRUE(array2[5]);
  EXPECT_TRUE(array2[6]);
  EXPECT_FALSE(array2[7]);
  EXPECT_FALSE(array2[8]);

  array2.free();
  array.free();

  derived.free();
  otherDerived.free();
}

GPU_TEST(managed_ptr, gpu_gpu_pointer_constructor)
{
  TestDerived* gpuPointer = chai::make_on_device<TestDerived>(3);
  chai::managed_ptr<TestDerived> derived({chai::GPU}, {gpuPointer});

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_FALSE(derived);
  EXPECT_TRUE(derived == nullptr);
  EXPECT_TRUE(nullptr == derived);
  EXPECT_FALSE(derived != nullptr);
  EXPECT_FALSE(nullptr != derived);

  chai::ManagedArray<int> array1(1, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(1, chai::GPU);
  chai::ManagedArray<bool> array3(5, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array1[i] = derived->getValue();
    array2[i] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
  });

  array1.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array1[0], 3);
  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);

  array3.free();
  array2.free();
  array1.free();

  derived.free();
}

GPU_TEST(managed_ptr, gpu_new_and_delete_on_device)
{
  // Initialize host side memory to hold a pointer
  Simple** cpuPointerHolder = (Simple**) malloc(sizeof(Simple*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  Simple** gpuPointerHolder = nullptr;
  chai::gpuMalloc((void**)(&gpuPointerHolder), sizeof(Simple*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  chai::gpuMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(Simple*), gpuMemcpyDeviceToHost);

  // Free device side memory
  chai::gpuFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  Simple* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::detail::destroy_on_device<<<1, 1>>>(gpuPointer);
}

GPU_TEST(managed_ptr, gpu_new_and_delete_on_device_2)
{
  // Initialize host side memory to hold a pointer
  Simple** cpuPointerHolder = (Simple**) malloc(sizeof(Simple*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  Simple** gpuPointerHolder = nullptr;
  chai::gpuMalloc((void**)(&gpuPointerHolder), sizeof(Simple*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  chai::gpuMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(Simple*), gpuMemcpyDeviceToHost);

  // Free device side memory
  chai::gpuFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  Simple* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::managed_ptr<Simple> test({chai::GPU}, {gpuPointer});
  test.free();
}

GPU_TEST(managed_ptr, simple_gpu_cpu_and_gpu_pointer_constructor)
{
  Simple* gpuPointer = chai::make_on_device<Simple>(3);
  Simple* cpuPointer = new Simple(4);

  chai::managed_ptr<Simple> simple({chai::GPU, chai::CPU}, {gpuPointer, cpuPointer});

  EXPECT_EQ(simple->getValue(), 4);

  chai::ManagedArray<int> array1(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array1[i] = simple->getValue();
  });

  array1.move(chai::CPU);

  EXPECT_EQ(array1[0], 3);

  array1.free();

  simple.free();
}

GPU_TEST(managed_ptr, gpu_cpu_and_gpu_pointer_constructor)
{
  TestDerived* gpuPointer = chai::make_on_device<TestDerived>(3);
  TestDerived* cpuPointer = new TestDerived(4);

  chai::managed_ptr<TestDerived> derived({chai::GPU, chai::CPU}, {gpuPointer, cpuPointer});

  EXPECT_EQ(derived->getValue(), 4);
  EXPECT_NE(derived.get(), nullptr);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);

  chai::ManagedArray<int> array1(1, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(1, chai::GPU);
  chai::ManagedArray<bool> array3(5, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array1[i] = derived->getValue();
    array2[i] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
  });

  array1.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array1[0], 3);
  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);

  array3.free();
  array2.free();
  array1.free();

  derived.free();
}

GPU_TEST(managed_ptr, gpu_make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);

  chai::ManagedArray<int> array(1, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(1, chai::GPU);
  chai::ManagedArray<bool> array3(7, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
    array2[i] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array[0], expectedValue);

  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);

  array3.free();
  array2.free();
  array.free();

  derived.free();
}

GPU_TEST(managed_ptr, gpu_allocate_managed_uses_allocator)
{
  auto& resourceManager = umpire::ResourceManager::getInstance();
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  auto deviceResource = resourceManager.getAllocator("HOST");
#else
  auto deviceResource = resourceManager.getAllocator("DEVICE");
#endif
  static auto deviceAllocator =
      resourceManager.makeAllocator<umpire::strategy::NamingShim>(
          "chai_managed_ptr_device_allocator", deviceResource);
  auto hostAllocator = resourceManager.getAllocator("HOST");
  const auto allocationsBefore = deviceAllocator.getAllocationCount();
  const int expectedValue = rand();

  chai::ManagedArray<int> destructionCounts(1, chai::CPU);
  destructionCounts[0] = 0;
  destructionCounts.move(chai::GPU);

  auto derived = chai::allocate_managed<DeviceAllocatorDerived>(
      {chai::CPU, chai::GPU}, {hostAllocator, deviceAllocator},
      expectedValue, chai::unpack(destructionCounts));
  chai::managed_ptr<DeviceAllocatorBase> base = derived;

  EXPECT_EQ(deviceAllocator.getAllocationCount(), allocationsBefore + 1);

  chai::ManagedArray<int> result(1, chai::GPU);
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    result[i] = base->getValue();
  });
  result.move(chai::CPU);
  EXPECT_EQ(result[0], expectedValue);
  result.free();

  base.free();

  int gpuDestructionCount = 0;
  chai::gpuMemcpy(&gpuDestructionCount,
                  destructionCounts.data(chai::GPU, false),
                  sizeof(int), gpuMemcpyDeviceToHost);
  EXPECT_EQ(destructionCounts.data(chai::CPU, false)[0], 1);
  EXPECT_EQ(gpuDestructionCount, 1);
  EXPECT_EQ(deviceAllocator.getAllocationCount(), allocationsBefore);

  destructionCounts.free();
}

GPU_TEST(managed_ptr, gpu_copy_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived(derived);

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
    array2[0] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
    array3[5] = derived == otherDerived;
    array3[6] = derived != otherDerived;

    array[1] = otherDerived->getValue();
    array2[1] = otherDerived.get();
    array3[7] = (bool) derived;
    array3[8] = derived == nullptr;
    array3[9] = nullptr == derived;
    array3[10] = derived != nullptr;
    array3[11] = nullptr != derived;
    array3[12] = derived == otherDerived;
    array3[13] = derived != otherDerived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array[0], expectedValue);
  EXPECT_EQ(array[1], expectedValue);

  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);
  EXPECT_TRUE(array3[5]);
  EXPECT_FALSE(array3[6]);

  EXPECT_NE(array2[1], nullptr);
  EXPECT_TRUE(array3[7]);
  EXPECT_FALSE(array3[8]);
  EXPECT_FALSE(array3[9]);
  EXPECT_TRUE(array3[10]);
  EXPECT_TRUE(array3[11]);
  EXPECT_TRUE(array3[12]);
  EXPECT_FALSE(array3[13]);

  array3.free();
  array2.free();
  array.free();

  derived.free();
}

GPU_TEST(managed_ptr, gpu_converting_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestBase> base(derived);

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestBase*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
    array2[0] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
    array3[5] = derived == base;
    array3[6] = derived != base;

    array[1] = base->getValue();
    array2[1] = base.get();
    array3[7] = (bool) base;
    array3[8] = base == nullptr;
    array3[9] = nullptr == base;
    array3[10] = base != nullptr;
    array3[11] = nullptr != base;
    array3[12] = base == derived;
    array3[13] = base != derived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array[0], expectedValue);
  EXPECT_EQ(array[1], expectedValue);

  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);
  EXPECT_TRUE(array3[5]);
  EXPECT_FALSE(array3[6]);

  EXPECT_NE(array2[1], nullptr);
  EXPECT_TRUE(array3[7]);
  EXPECT_FALSE(array3[8]);
  EXPECT_FALSE(array3[9]);
  EXPECT_TRUE(array3[10]);
  EXPECT_TRUE(array3[11]);
  EXPECT_TRUE(array3[12]);
  EXPECT_FALSE(array3[13]);

  array3.free();
  array2.free();
  array.free();

  base.free();
}

GPU_TEST(managed_ptr, gpu_copy_assignment_operator)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
    array2[0] = derived.get();
    array3[0] = (bool) derived;
    array3[1] = derived == nullptr;
    array3[2] = nullptr == derived;
    array3[3] = derived != nullptr;
    array3[4] = nullptr != derived;
    array3[5] = derived == otherDerived;
    array3[6] = derived != otherDerived;

    array[1] = otherDerived->getValue();
    array2[1] = otherDerived.get();
    array3[7] = (bool) derived;
    array3[8] = derived == nullptr;
    array3[9] = nullptr == derived;
    array3[10] = derived != nullptr;
    array3[11] = nullptr != derived;
    array3[12] = derived == otherDerived;
    array3[13] = derived != otherDerived;
  });

  array.move(chai::CPU);
  array2.move(chai::CPU);
  array3.move(chai::CPU);

  EXPECT_EQ(array[0], expectedValue);
  EXPECT_EQ(array[1], expectedValue);

  EXPECT_NE(array2[0], nullptr);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);
  EXPECT_TRUE(array3[5]);
  EXPECT_FALSE(array3[6]);

  EXPECT_NE(array2[1], nullptr);
  EXPECT_TRUE(array3[7]);
  EXPECT_FALSE(array3[8]);
  EXPECT_FALSE(array3[9]);
  EXPECT_TRUE(array3[10]);
  EXPECT_TRUE(array3[11]);
  EXPECT_TRUE(array3[12]);
  EXPECT_FALSE(array3[13]);

  array3.free();
  array2.free();
  array.free();

  derived.free();
}

#endif
