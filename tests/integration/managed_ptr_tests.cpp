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

#include "../src/util/forall.hpp"

#include "chai/managed_ptr.hpp"

// Standard library headers
#include <cstdlib>

class TestBase {
   public:
      TestBase() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const = 0;
};

class TestDerived : public TestBase {
   public:
      TestDerived() : TestBase(), m_values(nullptr) {}
      TestDerived(chai::ManagedArray<int> values) : TestBase(), m_values(values) {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const { return m_values[i]; }

   private:
      chai::ManagedArray<int> m_values;
};

class TestInnerBase {
   public:
      TestInnerBase() {}

      CHAI_HOST_DEVICE virtual int getValue() = 0;
};

class TestInner : public TestInnerBase {
   public:
      TestInner() : TestInnerBase(), m_value(0) {}
      TestInner(int value) : TestInnerBase(), m_value(value) {}

      CHAI_HOST_DEVICE virtual int getValue() { return m_value; }

   private:
      int m_value;
};

class TestContainer {
   public:
      TestContainer() : m_innerType(nullptr) {}
      TestContainer(chai::managed_ptr<TestInnerBase> innerType) : m_innerType(innerType) {}

      CHAI_HOST_DEVICE virtual int getValue() const {
         return m_innerType->getValue();
      }

   private:
      chai::managed_ptr<TestInnerBase> m_innerType;
};

TEST(managed_ptr, inner_ManagedArray)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  chai::managed_ptr<TestDerived> derived(new TestDerived(array));
  ASSERT_EQ(derived->getValue(0), expectedValue);
}

CUDA_TEST(managed_ptr, cuda_inner_ManagedArray)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  chai::managed_ptr<TestBase> derived(new TestDerived(array));
  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

TEST(managed_ptr, inner_managed_ptr)
{
  const int expectedValue = rand();

  chai::managed_ptr<TestInner> derived(new TestInner(expectedValue));
  TestContainer container(derived);

  ASSERT_EQ(container.getValue(), expectedValue);
}

CUDA_TEST(managed_ptr, cuda_inner_managed_ptr)
{
  const int expectedValue = rand();

  chai::managed_ptr<TestInner> derived(new TestInner(expectedValue));
  TestContainer container(derived);

  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    results[i] = container.getValue();
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);
}

