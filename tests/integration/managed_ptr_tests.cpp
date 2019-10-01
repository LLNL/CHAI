// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
// rights reserved.
//
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of CHAI.
//
// LLNL-CODE-705877
//
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()

#include "chai/config.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"

#include "../src/util/forall.hpp"

// Standard library headers
#include <cstdlib>

class Base1 {
   public:
      CHAI_HOST_DEVICE Base1() {}
      CHAI_HOST_DEVICE virtual ~Base1() {}

      CHAI_HOST_DEVICE virtual bool isBase1() { return true; }
};

class Base2 {
   public:
      CHAI_HOST_DEVICE Base2() {}
      CHAI_HOST_DEVICE virtual ~Base2() {}

      CHAI_HOST_DEVICE virtual bool isBase2() { return true; }
};

class ClassWithMultipleInheritance : public Base1, public Base2 {
   public:
      CHAI_HOST_DEVICE ClassWithMultipleInheritance() : Base1(), Base2() {}
      CHAI_HOST_DEVICE virtual ~ClassWithMultipleInheritance() {}
};

class RawArrayClass {
   public:
      CHAI_HOST_DEVICE RawArrayClass() : m_values(nullptr) {}
      CHAI_HOST_DEVICE RawArrayClass(int* values) : m_values(values) {}

      CHAI_HOST_DEVICE ~RawArrayClass() {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_values[i]; }

   private:
      int* m_values;
};

class RawPointerClass {
   public:
      CHAI_HOST_DEVICE RawPointerClass() : m_innerClass(nullptr) {}
      CHAI_HOST_DEVICE RawPointerClass(RawArrayClass* innerClass) : m_innerClass(innerClass) {}

      CHAI_HOST_DEVICE ~RawPointerClass() {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_innerClass->getValue(i); }

   private:
      RawArrayClass* m_innerClass;
};

class TestBase {
   public:
      CHAI_HOST_DEVICE TestBase() {}
      CHAI_HOST_DEVICE virtual ~TestBase() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived() : TestBase(), m_values(nullptr) {}
      CHAI_HOST_DEVICE TestDerived(chai::ManagedArray<int> values) : TestBase(), m_values(values) {}
      CHAI_HOST_DEVICE virtual ~TestDerived() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const { return m_values[i]; }

   private:
      chai::ManagedArray<int> m_values;
};

class TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInnerBase() {}
      CHAI_HOST_DEVICE virtual ~TestInnerBase() {}

      CHAI_HOST_DEVICE virtual int getValue() = 0;
};

class TestInner : public TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInner() : TestInnerBase(), m_value(0) {}
      CHAI_HOST_DEVICE TestInner(int value) : TestInnerBase(), m_value(value) {}
      CHAI_HOST_DEVICE virtual ~TestInner() {}

      CHAI_HOST_DEVICE virtual int getValue() { return m_value; }

   private:
      int m_value;
};

class TestContainer {
   public:
      CHAI_HOST_DEVICE TestContainer() : m_innerType(nullptr) {}
      CHAI_HOST_DEVICE TestContainer(chai::managed_ptr<TestInner> innerType) : m_innerType(innerType) {}

      CHAI_HOST_DEVICE ~TestContainer() {}

      CHAI_HOST_DEVICE int getValue() const {
         return m_innerType->getValue();
      }

   private:
      chai::managed_ptr<TestInner> m_innerType;
};

class MultipleRawArrayClass {
   public:
      CHAI_HOST_DEVICE MultipleRawArrayClass() : m_values1(nullptr), m_values2(nullptr) {}
      CHAI_HOST_DEVICE MultipleRawArrayClass(int* values1, int* values2) :
         m_values1(values1),
         m_values2(values2)
      {}

      CHAI_HOST_DEVICE ~MultipleRawArrayClass() {}

      CHAI_HOST_DEVICE int getValue(const int i, const int j) const {
         if (i == 0) {
            return m_values1[j];
         }
         else if (i == 1) {
            return m_values2[j];
         }
         else {
            return -1;
         }
      }

   private:
      int* m_values1;
      int* m_values2;
};

TEST(managed_ptr, class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
    array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);

  ASSERT_EQ(rawArrayClass->getValue(0), expectedValue);

  array.free();
}

TEST(managed_ptr, class_with_multiple_raw_arrays)
{
  const int expectedValue1 = rand();
  const int expectedValue2 = rand();

  chai::ManagedArray<int> array1(1, chai::CPU);
  chai::ManagedArray<int> array2(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array1[i] = expectedValue1;
     array2[i] = expectedValue2;
  });

  auto multipleRawArrayClass = chai::make_managed<MultipleRawArrayClass>(array1, array2);

  ASSERT_EQ(multipleRawArrayClass->getValue(0, 0), expectedValue1);
  ASSERT_EQ(multipleRawArrayClass->getValue(1, 0), expectedValue2);
}

TEST(managed_ptr, class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);

  ASSERT_EQ(derived->getValue(0), expectedValue);
}

TEST(managed_ptr, class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  auto rawPointerClass = chai::make_managed<RawPointerClass>(rawArrayClass);

  // This prevents the pointers contained by rawArrayClass from being deleted
  // out from under us. Otherwise, rawArrayClass is the last remaining reference
  // and if it is destroyed before rawPointerClass is, then we are in trouble.
  rawPointerClass.set_callback([=] (chai::Action, chai::ExecutionSpace, void*) {
                                  (void) rawArrayClass; return false;
                               });
  rawArrayClass = nullptr;

  ASSERT_EQ((*rawPointerClass).getValue(0), expectedValue);
}

TEST(managed_ptr, class_with_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  TestContainer container(derived);

  ASSERT_EQ(container.getValue(), expectedValue);
}

TEST(managed_ptr, nested_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  auto container = chai::make_managed<TestContainer>(derived);

  ASSERT_EQ(container->getValue(), expectedValue);
}

#ifdef __CUDACC__

template <typename T>
__global__ void deviceNew(T** arr) {
   *arr = new T[5];
}

template <typename T>
__global__ void deviceDelete(T** arr) {
   delete[] *arr;
}

__global__ void passObjectToKernel(chai::ManagedArray<int> arr) {
   arr[0] = -1;
}

CUDA_TEST(managed_ptr, make_on_device)
{
  int** hostArray = (int**) malloc(sizeof(int*));
  hostArray[0] = nullptr;

  int** deviceArray = nullptr;
  cudaMalloc(&deviceArray, sizeof(int*));

  int** deviceArray2 = nullptr;
  cudaMalloc(&deviceArray2, sizeof(int*));

  deviceNew<<<1, 1>>>(deviceArray);

  cudaMemcpy(hostArray, deviceArray, sizeof(int*), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaMemcpy(deviceArray2, hostArray, sizeof(int*), cudaMemcpyHostToDevice);
  ASSERT_NE(hostArray[0], nullptr);

  deviceDelete<<<1, 1>>>(deviceArray2);
  cudaDeviceSynchronize();
  free(hostArray);
  cudaFree(deviceArray);
  cudaFree(deviceArray2);
}

CUDA_TEST(managed_ptr, cuda_new_and_delete_on_device)
{
  // Initialize host side memory to hold a pointer
  RawArrayClass** cpuPointerHolder = (RawArrayClass**) malloc(sizeof(RawArrayClass*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  RawArrayClass** gpuPointerHolder = nullptr;
  cudaMalloc(&gpuPointerHolder, sizeof(RawArrayClass*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  cudaMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(RawArrayClass*), cudaMemcpyDeviceToHost);

  // Free device side memory
  cudaFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  RawArrayClass* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::detail::destroy_on_device<<<1, 1>>>(gpuPointer);
}

CUDA_TEST(managed_ptr, cuda_build_managed_ptr)
{
  // Initialize host side memory to hold a pointer
  RawArrayClass** cpuPointerHolder = (RawArrayClass**) malloc(sizeof(RawArrayClass*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  RawArrayClass** gpuPointerHolder = nullptr;
  cudaMalloc(&gpuPointerHolder, sizeof(RawArrayClass*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  cudaMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(RawArrayClass*), cudaMemcpyDeviceToHost);

  // Free device side memory
  cudaFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  RawArrayClass* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::managed_ptr<RawArrayClass> managedPtr({chai::GPU}, {gpuPointer});
}


CUDA_TEST(managed_ptr, pass_object_to_kernel)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  chai::ArrayManager* manager = chai::ArrayManager::getInstance();
  manager->setExecutionSpace(chai::GPU);
  passObjectToKernel<<<1, 1>>>(array);
  cudaDeviceSynchronize();
  array.move(chai::CPU);
  cudaDeviceSynchronize();
  ASSERT_EQ(array[0], -1);
}

CUDA_TEST(managed_ptr, cuda_class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = rawArrayClass->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_raw_array_and_callback)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto cpuPointer = new RawArrayClass(array);
  auto gpuPointer = chai::detail::make_on_device<RawArrayClass>(array);

  auto callback = [=] (chai::Action action, chai::ExecutionSpace space, void*) mutable -> bool {
     switch (action) {
        case chai::ACTION_FREE:
           switch (space) {
              case chai::NONE:
                 array.free();
                 return true;
              default:
                 return false;
           }
        default:
           return false;
     }
  };

  auto managedPointer = chai::managed_ptr<RawArrayClass>({chai::CPU, chai::GPU},
                                                         {cpuPointer, gpuPointer},
                                                         callback);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = managedPointer->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(array);

  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  auto rawPointerClass = chai::make_managed<RawPointerClass>(rawArrayClass);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = (*rawPointerClass).getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  TestContainer container(derived);

  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = container.getValue();
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_nested_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  auto container = chai::make_managed<TestContainer>(derived);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = container->getValue();
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

CUDA_TEST(managed_ptr, cuda_multiple_inheritance)
{
  auto derived = chai::make_managed<ClassWithMultipleInheritance>();

  chai::managed_ptr<Base1> base1 = derived;
  chai::managed_ptr<Base2> base2 = derived;

  chai::ManagedArray<bool> results(2, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = base1->isBase1();
    results[1] = base2->isBase2();
  });

  results.move(chai::CPU);
  cudaDeviceSynchronize();

  ASSERT_EQ(results[0], true);
  ASSERT_EQ(results[1], true);
}

CUDA_TEST(managed_ptr, static_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::static_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::static_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);
}

CUDA_TEST(managed_ptr, dynamic_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::dynamic_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::dynamic_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);
}

CUDA_TEST(managed_ptr, const_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto constDerived = chai::const_pointer_cast<const TestDerived>(derived);
  auto derivedFromConst = chai::const_pointer_cast<TestDerived>(constDerived);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = constDerived->getValue(i);
    results[2] = derivedFromConst->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);
}

CUDA_TEST(managed_ptr, reinterpret_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::reinterpret_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::reinterpret_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);
}

#endif

#if 0 // TODO: Enable if/when ManagedArrays of managed_ptrs can be handled correctly.

class RawArrayOfPointersClass {
   public:
      CHAI_HOST_DEVICE RawArrayOfPointersClass() = delete;
      CHAI_HOST_DEVICE RawArrayOfPointersClass(RawArrayClass** arrayOfPointers) :
         m_arrayOfPointers(arrayOfPointers)
      {}

      CHAI_HOST_DEVICE int getValue(const int i, const int j) const {
         return m_arrayOfPointers[i]->getValue(j);
      }

   private:
      RawArrayClass** m_arrayOfPointers = nullptr;
};

TEST(managed_ptr, class_with_raw_array_of_pointers)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  chai::managed_ptr<RawArrayClass> arrayOfPointers[1] = {rawArrayClass};

  auto rawArrayOfPointersClass = chai::make_managed<RawArrayOfPointersClass>(arrayOfPointers);
  ASSERT_EQ(rawArrayOfPointersClass->getValue(0, 0), expectedValue);
}

#endif

