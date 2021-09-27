//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/managed_ptr.hpp"
#include "../src/util/forall.hpp"
#include <cstdio>

class TestBase {
   public:
      CHAI_HOST_DEVICE TestBase() {}
      CHAI_HOST_DEVICE virtual ~TestBase() {}
      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived(const int value) : TestBase(), m_value(value) {}
      CHAI_HOST_DEVICE virtual ~TestDerived() {}
      CHAI_HOST_DEVICE virtual int getValue() const { return m_value; }

   private:
      int m_value;
};

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{
  // Create an object on the host, and if either CUDA or HIP is enabled,
  // on the device as well
  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(42);

  // chai::managed_ptr can be accessed on the host
  forall(sequential(), 0, 1, [=] CHAI_HOST (int i) {
    printf("Result of virtual function call on host: %d\n", derived->getValue());
  });

#if defined(CHAI_GPUCC)
  // chai::managed_ptr can be accessed on the device
  forall(gpu(), 0, 1, [=] CHAI_DEVICE (int i) {
    printf("Result of virtual function call on device: %d\n", derived->getValue());
  });
#endif

  // Free the object on the host, and if either CUDA or HIP is enabled,
  // on the device as well
  derived.free();

  return 0;
}

