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

__global__ void callDeviceVirtualFunction(int* d_result,
                                           chai::managed_ptr<TestBase> derived) {
   *d_result = derived->getValue();
}

int main(int, char**) {
   chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(42);

   int* d_result;
   hipMalloc(&d_result, sizeof(int));

   hipLaunchKernelGGL(callDeviceVirtualFunction, 1, 1, 0, 0, d_result, derived);

   int* h_result = (int*) malloc(sizeof(int));
   hipMemcpyDtoH(h_result, d_result, sizeof(int));

   printf("Result: %d\n", *h_result);

   free(h_result);
   hipFree(d_result);
   derived.free();

   return 0;
}

