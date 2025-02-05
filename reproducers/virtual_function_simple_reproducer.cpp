//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/managed_ptr.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>

class TestBase {
   public:
      __host__ __device__ TestBase() {}
      __host__ __device__ virtual ~TestBase() {}
      __host__ __device__ virtual int getValue() const = 0;
};

class TestDerived : public TestBase {
   public:
      __host__ __device__ TestDerived(const int value) : TestBase(), m_value(value) {}
      __host__ __device__ virtual ~TestDerived() {}
      __host__ __device__ virtual int getValue() const { return m_value; }

   private:
      int m_value;
};

__global__ void callVirtualFunction() {
   TestBase* derived = new TestDerived(42);
   int result = derived->getValue();
   (void) result;
   return;
}

int main(int, char**) {
   hipLaunchKernelGGL(callVirtualFunction, 1, 1, 0, 0);
   return 0;
}

