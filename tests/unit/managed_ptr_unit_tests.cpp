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
      CHAI_HOST_DEVICE TestBase() {}

      CHAI_HOST_DEVICE static TestBase* Factory(const int value);

      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived() : TestBase(), m_value(0) {}
      CHAI_HOST_DEVICE TestDerived(const int value) : TestBase(), m_value(value) {}

      CHAI_HOST_DEVICE virtual int getValue() const { return m_value; }

   private:
      int m_value;
};

CHAI_HOST_DEVICE TestBase* TestBase::Factory(const int value) {
   return new TestDerived(value);
}

CHAI_HOST_DEVICE TestBase* Factory(const int value) {
   return new TestDerived(value);
}

CHAI_HOST_DEVICE TestBase* OverloadedFactory() {
   return new TestDerived(-1);
}

CHAI_HOST_DEVICE TestBase* OverloadedFactory(const int value) {
   return new TestDerived(value);
}

TEST(managed_ptr, default_constructor)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_FALSE(derived);
  EXPECT_TRUE(derived == nullptr);
  EXPECT_TRUE(nullptr == derived);
  EXPECT_FALSE(derived != nullptr);
  EXPECT_FALSE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(derived != otherDerived);
  EXPECT_FALSE(otherDerived != derived);
}

CUDA_TEST(managed_ptr, cuda_default_constructor)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;

  chai::ManagedArray<TestDerived*> array(1, chai::GPU);
  chai::ManagedArray<bool> array2(9, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
}

TEST(managed_ptr, nullptr_constructor)
{
  chai::managed_ptr<TestDerived> derived = nullptr;
  chai::managed_ptr<TestDerived> otherDerived = nullptr;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_FALSE(derived);
  EXPECT_TRUE(derived == nullptr);
  EXPECT_TRUE(nullptr == derived);
  EXPECT_FALSE(derived != nullptr);
  EXPECT_FALSE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(derived != otherDerived);
  EXPECT_FALSE(otherDerived != derived);
}

CUDA_TEST(managed_ptr, cuda_nullptr_constructor)
{
  chai::managed_ptr<TestDerived> derived = nullptr;
  chai::managed_ptr<TestDerived> otherDerived = nullptr;

  chai::ManagedArray<TestDerived*> array(1, chai::GPU);
  chai::ManagedArray<bool> array2(7, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
}

TEST(managed_ptr, make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
}

CUDA_TEST(managed_ptr, cuda_make_managed)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);

  chai::ManagedArray<int> array(1, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(1, chai::GPU);
  chai::ManagedArray<bool> array3(7, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(array3[0]);
  EXPECT_FALSE(array3[1]);
  EXPECT_FALSE(array3[2]);
  EXPECT_TRUE(array3[3]);
  EXPECT_TRUE(array3[4]);
}

TEST(managed_ptr, make_managed_from_factory_function)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed_from_factory<TestBase>(Factory, expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
}

CUDA_TEST(managed_ptr, make_managed_from_factory_lambda)
{
  const int expectedValue = rand();

  auto factory = [] CHAI_HOST_DEVICE (const int value) {
    return new TestDerived(value);
  };

  auto derived = chai::make_managed_from_factory<TestBase>(factory, expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
}

CUDA_TEST(managed_ptr, make_managed_from_overloaded_factory_function)
{
  const int expectedValue = rand();

  auto factory = [] CHAI_HOST_DEVICE (const int value) {
    return OverloadedFactory(value);
  };

  auto derived = chai::make_managed_from_factory<TestBase>(factory, expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
}

TEST(managed_ptr, make_managed_from_factory_static_member_function)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed_from_factory<TestBase>(&TestBase::Factory, expectedValue);

  EXPECT_EQ((*derived).getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 1);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
}

TEST(managed_ptr, copy_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived(derived);

  EXPECT_EQ(derived->getValue(), expectedValue);
  EXPECT_EQ(otherDerived->getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_FALSE(derived != otherDerived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 2);
  EXPECT_TRUE(otherDerived);
  EXPECT_FALSE(otherDerived == nullptr);
  EXPECT_FALSE(nullptr == otherDerived);
  EXPECT_TRUE(otherDerived != nullptr);
  EXPECT_TRUE(nullptr != otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(otherDerived != derived);
}

CUDA_TEST(managed_ptr, cuda_copy_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived(derived);

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
}

TEST(managed_ptr, converting_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestBase> base = derived;

  EXPECT_EQ(derived->getValue(), expectedValue);
  EXPECT_EQ(base->getValue(), expectedValue);

  EXPECT_NE(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == base);
  EXPECT_FALSE(derived != base);

  EXPECT_NE(base.get(), nullptr);
  EXPECT_EQ(base.use_count(), 2);
  EXPECT_TRUE(base);
  EXPECT_FALSE(base == nullptr);
  EXPECT_FALSE(nullptr == base);
  EXPECT_TRUE(base != nullptr);
  EXPECT_TRUE(nullptr != base);
  EXPECT_TRUE(base == derived);
  EXPECT_FALSE(base != derived);
}

CUDA_TEST(managed_ptr, cuda_converting_constructor)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestBase> base(derived);

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestBase*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_TRUE(derived);
  EXPECT_FALSE(derived == nullptr);
  EXPECT_FALSE(nullptr == derived);
  EXPECT_TRUE(derived != nullptr);
  EXPECT_TRUE(nullptr != derived);
  EXPECT_TRUE(derived == otherDerived);
  EXPECT_FALSE(derived != otherDerived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 2);
  EXPECT_TRUE(otherDerived);
  EXPECT_FALSE(otherDerived == nullptr);
  EXPECT_FALSE(nullptr == otherDerived);
  EXPECT_TRUE(otherDerived != nullptr);
  EXPECT_TRUE(nullptr != otherDerived);
  EXPECT_TRUE(otherDerived == derived);
  EXPECT_FALSE(otherDerived != derived);
}

CUDA_TEST(managed_ptr, cuda_copy_assignment_operator)
{
  const int expectedValue = rand();
  auto derived = chai::make_managed<TestDerived>(expectedValue);
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  chai::ManagedArray<int> array(2, chai::GPU);
  chai::ManagedArray<TestDerived*> array2(2, chai::GPU);
  chai::ManagedArray<bool> array3(14, chai::GPU);
  
  forall(cuda(), 0, 1, [=] __device__ (int i) {
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
}

TEST(managed_ptr, copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived(derived);

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 0);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestDerived> otherDerived;
  otherDerived = derived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 0);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, conversion_copy_constructor_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived(derived);

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 0);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);
}

TEST(managed_ptr, conversion_copy_assignment_operator_from_default_constructed)
{
  chai::managed_ptr<TestDerived> derived;
  chai::managed_ptr<TestBase> otherDerived;
  otherDerived = derived;

  EXPECT_EQ(derived.get(), nullptr);
  EXPECT_EQ(derived.use_count(), 0);
  EXPECT_EQ(bool(derived), false);
  EXPECT_EQ(derived, nullptr);
  EXPECT_EQ(nullptr, derived);

  EXPECT_EQ(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 0);
  EXPECT_EQ(bool(otherDerived), false);
  EXPECT_EQ(otherDerived, nullptr);
  EXPECT_EQ(nullptr, otherDerived);
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
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_EQ(bool(derived), true);
  EXPECT_NE(derived, nullptr);
  EXPECT_NE(nullptr, derived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 1);
  EXPECT_EQ(bool(otherDerived), true);
  EXPECT_NE(otherDerived, nullptr);
  EXPECT_NE(nullptr, otherDerived);

  EXPECT_NE(thirdDerived.get(), nullptr);
  EXPECT_EQ(thirdDerived.use_count(), 2);
  EXPECT_EQ(bool(thirdDerived), true);
  EXPECT_NE(thirdDerived, nullptr);
  EXPECT_NE(nullptr, thirdDerived);
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
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_EQ(bool(derived), true);
  EXPECT_NE(derived, nullptr);
  EXPECT_NE(nullptr, derived);

  EXPECT_NE(otherDerived.get(), nullptr);
  EXPECT_EQ(otherDerived.use_count(), 1);
  EXPECT_EQ(bool(otherDerived), true);
  EXPECT_NE(otherDerived, nullptr);
  EXPECT_NE(nullptr, otherDerived);

  EXPECT_NE(thirdDerived.get(), nullptr);
  EXPECT_EQ(thirdDerived.use_count(), 2);
  EXPECT_EQ(bool(thirdDerived), true);
  EXPECT_NE(thirdDerived, nullptr);
  EXPECT_NE(nullptr, thirdDerived);
}

