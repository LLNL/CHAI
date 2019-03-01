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

class RawArrayClass {
   public:
      CHAI_HOST_DEVICE RawArrayClass() : m_values(nullptr) {}
      CHAI_HOST_DEVICE RawArrayClass(int* values) : m_values(values) {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_values[i]; }

   private:
      int* m_values;
};

class RawPointerClass {
   public:
      CHAI_HOST_DEVICE RawPointerClass() : m_innerClass(nullptr) {}
      CHAI_HOST_DEVICE RawPointerClass(RawArrayClass* innerClass) : m_innerClass(innerClass) {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_innerClass->getValue(i); }

   private:
      RawArrayClass* m_innerClass;
};

class TestBase {
   public:
      CHAI_HOST_DEVICE TestBase() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived() : TestBase(), m_values(nullptr) {}
      CHAI_HOST_DEVICE TestDerived(chai::ManagedArray<int> values) : TestBase(), m_values(values) {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const { return m_values[i]; }

   private:
      chai::ManagedArray<int> m_values;
};

class TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInnerBase() {}

      CHAI_HOST_DEVICE virtual int getValue() = 0;
};

class TestInner : public TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInner() : TestInnerBase(), m_value(0) {}
      CHAI_HOST_DEVICE TestInner(int value) : TestInnerBase(), m_value(value) {}

      CHAI_HOST_DEVICE virtual int getValue() { return m_value; }

   private:
      int m_value;
};

class TestContainer {
   public:
      CHAI_HOST_DEVICE TestContainer() : m_innerType(nullptr) {}
      CHAI_HOST_DEVICE TestContainer(chai::managed_ptr<TestInner> innerType) : m_innerType(innerType) {}

      CHAI_HOST_DEVICE virtual int getValue() const {
         return m_innerType->getValue();
      }

   private:
      chai::managed_ptr<TestInner> m_innerType;
};

TEST(managed_ptr, class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  ASSERT_EQ(rawArrayClass->getValue(0), expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = rawArrayClass->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

TEST(managed_ptr, class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto derived = chai::make_managed<TestDerived>(array);
  ASSERT_EQ(derived->getValue(0), expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(array);
  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

TEST(managed_ptr, class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  auto rawPointerClass = chai::make_managed<RawPointerClass>(rawArrayClass);

  ASSERT_EQ((*rawPointerClass).getValue(0), expectedValue);
}

CUDA_TEST(managed_ptr, cuda_class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(array);
  auto rawPointerClass = chai::make_managed<RawPointerClass>(rawArrayClass);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = (*rawPointerClass).getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

TEST(managed_ptr, class_with_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  TestContainer container(derived);

  ASSERT_EQ(container.getValue(), expectedValue);
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

TEST(managed_ptr, nested_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  auto container = chai::make_managed<TestContainer>(derived);

  ASSERT_EQ(container->getValue(), expectedValue);
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

