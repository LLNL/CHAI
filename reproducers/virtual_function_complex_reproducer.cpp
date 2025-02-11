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

__global__ void createOnDevice(TestDerived** ptr, int val) {
   *ptr = new TestDerived(val);
}

__global__ void destroyOnDevice(TestBase* ptr) {
   delete ptr;
}

__global__ void callDeviceVirtualFunction(int* d_result, TestBase* derived) {
   *d_result = derived->getValue();
}

int main(int, char**) {
   // Allocate space on the GPU to hold the pointer to the new object
   TestDerived** gpuBuffer;
   hipMalloc((void**)(&gpuBuffer), sizeof(TestDerived*));

   // New the object on the device
   hipLaunchKernelGGL(createOnDevice, 1, 1, 0, 0, gpuBuffer, 42);

   // Allocate space on the CPU for the GPU pointer and copy the pointer to the CPU
   TestDerived** cpuBuffer = (TestDerived**) malloc(sizeof(TestDerived*));
   hipMemcpyDtoH(cpuBuffer, gpuBuffer, sizeof(TestDerived*));

   // Extract the pointer to the object on the GPU
   TestBase* derived = *cpuBuffer;

   // Free the host and device buffers
   free(cpuBuffer);
   hipFree(gpuBuffer);

   // Allocate space to hold the result of calling a virtual function on the device
   int* d_result;
   hipMalloc(&d_result, sizeof(int));

   // Call a virtual function on the device
   hipLaunchKernelGGL(callDeviceVirtualFunction, 1, 1, 0, 0, d_result, derived);

   // Copy the result to the CPU
   int* h_result = (int*) malloc(sizeof(int));
   hipMemcpyDtoH(h_result, d_result, sizeof(int));

   // Print the result
   printf("Result: %d\n", *h_result);

   // Free memory
   free(h_result);
   hipFree(d_result);

   // Delete the object on the device
   hipLaunchKernelGGL(destroyOnDevice, 1, 1, 0, 0, derived);

   return 0;
}

