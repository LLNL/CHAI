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

      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class TestDerived : public TestBase {
   public:
      TestDerived() : TestBase(), m_value(0) {}
      TestDerived(const int value) : TestBase(), m_value(value) {}

      CHAI_HOST_DEVICE virtual int getValue() const { return m_value; }

   private:
      int m_value;
};

TEST(managed_ptr, DefaultConstructor)
{
  chai::managed_ptr<TestDerived> derived;
  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);
}

TEST(managed_ptr, nullptrConstructor)
{
  chai::managed_ptr<TestDerived> derived = nullptr;
  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);
}

TEST(managed_ptr, HostPtrConstructor)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestDerived> derived(new TestDerived(expectedValue));
  ASSERT_EQ(derived->getValue(), expectedValue);

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 1);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);
}

CUDA_TEST(managed_ptr, cuda_HostPtrConstructor)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestDerived> derived(new TestDerived(expectedValue));
  chai::ManagedArray<int> array(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
  });

  array.move(chai::CPU);
  ASSERT_EQ(array[0], expectedValue);
}

TEST(managed_ptr, ConvertingPtrConstructor)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestBase> derived(new TestDerived(expectedValue));
  ASSERT_EQ(derived->getValue(), expectedValue);

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 1);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);
}

CUDA_TEST(managed_ptr, cuda_ConvertingPtrConstructor)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestBase> derived(new TestDerived(expectedValue));
  chai::ManagedArray<int> array(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    array[i] = derived->getValue();
  });

  array.move(chai::CPU);
  ASSERT_EQ(array[0], expectedValue);
}

TEST(managed_ptr, make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  ASSERT_EQ((*derived).getValue(), expectedValue);

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 1);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);
}

CUDA_TEST(managed_ptr, cuda_make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::ManagedArray<int> array(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    array[i] = (*derived).getValue();
  });

  array.move(chai::CPU);
  ASSERT_EQ(array[0], expectedValue);
}

TEST(managed_ptr, converting_make_managed)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(expectedValue);
  ASSERT_EQ((*derived).getValue(), expectedValue);

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 1);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);
}

CUDA_TEST(managed_ptr, cuda_converting_make_managed)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(expectedValue);
  chai::ManagedArray<int> array(1, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
    array[i] = (*derived).getValue();
  });

  array.move(chai::CPU);
  ASSERT_EQ(array[0], expectedValue);
}

TEST(managed_ptr, copy_constructor)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestDerived> derived(new TestDerived(expectedValue));
  chai::managed_ptr<TestDerived> otherDerived(derived);

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 2);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);

  ASSERT_NE(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 2);
  ASSERT_EQ(bool(otherDerived), true);
  ASSERT_NE(otherDerived, nullptr);
  ASSERT_NE(nullptr, otherDerived);
}

TEST(managed_ptr, copy_assignment_operator)
{
  const int expectedValue = rand();
  chai::managed_ptr<TestDerived> derived(new TestDerived(expectedValue));
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  ASSERT_NE(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 2);
  ASSERT_EQ(bool(derived), true);
  ASSERT_NE(derived, nullptr);
  ASSERT_NE(nullptr, derived);

  ASSERT_NE(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 2);
  ASSERT_EQ(bool(otherDerived), true);
  ASSERT_NE(otherDerived, nullptr);
  ASSERT_NE(nullptr, otherDerived);
}

TEST(managed_ptr, copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived(derived);

  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);

  ASSERT_EQ(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 0);
  ASSERT_EQ(bool(otherDerived), false);
  ASSERT_EQ(otherDerived, nullptr);
  ASSERT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);

  ASSERT_EQ(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 0);
  ASSERT_EQ(bool(otherDerived), false);
  ASSERT_EQ(otherDerived, nullptr);
  ASSERT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, conversion_copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived(derived);

  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);

  ASSERT_EQ(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 0);
  ASSERT_EQ(bool(otherDerived), false);
  ASSERT_EQ(otherDerived, nullptr);
  ASSERT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, conversion_copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived;
  otherDerived = derived;

  ASSERT_EQ(derived.get(), nullptr);
  ASSERT_EQ(derived.use_count(), 0);
  ASSERT_EQ(bool(derived), false);
  ASSERT_EQ(derived, nullptr);
  ASSERT_EQ(nullptr, derived);

  ASSERT_EQ(otherDerived.get(), nullptr);
  ASSERT_EQ(otherDerived.use_count(), 0);
  ASSERT_EQ(bool(otherDerived), false);
  ASSERT_EQ(otherDerived, nullptr);
  ASSERT_EQ(nullptr, otherDerived);
}

