//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

class TestBase1 {
   public:
      __host__ __device__ TestBase1() {}
      __host__ __device__ virtual ~TestBase1() {}
      __host__ __device__ virtual int getValue1() const = 0;
};

class TestBase2 {
   public:
      __host__ __device__ TestBase2() {}
      __host__ __device__ virtual ~TestBase2() {}
      __host__ __device__ virtual int getValue2() const = 0;
};

class TestDerived : public TestBase1, public TestBase2 {
   public:
      __host__ __device__ TestDerived(int value1, int value2) :
         TestBase1(), TestBase2(), m_value1(value1), m_value2(value2) {}
      __host__ __device__ virtual ~TestDerived() {}
      __host__ __device__ int getValue1() const override { return m_value1; }
      __host__ __device__ int getValue2() const override { return m_value2; }

   private:
      int m_value1;
      int m_value2;
};

__global__ void callVirtualFunction() {
   TestDerived derived(1, 2);

   TestBase1* testBase1Ptr = &derived;
   int value1 = testBase1Ptr->getValue1();
   (void) value1;

   TestBase2* testBase2Ptr = &derived;
   int value2 = testBase2Ptr->getValue2();
   (void) value2;

   return;
}

int main(int, char**) {
   hipLaunchKernelGGL(callVirtualFunction, 1, 1, 0, 0);
   return 0;
}

