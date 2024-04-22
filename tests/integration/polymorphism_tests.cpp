//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/ChaiMacros.hpp"
#include "chai/ManagedSharedPtr.hpp"
#include "chai/SharedPtrManager.hpp"
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X##Y();    \
  TEST(X, Y) { gpu_test_##X##Y(); } \
  static void gpu_test_##X##Y()

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"
#include "chai/ManagedSharedPtr.hpp"

#include "../src/util/forall.hpp"

// Standard library headers
#include <cstdlib>

inline void gpuErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "[CHAI] GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}

#define GPU_ERROR_CHECK(code) { gpuErrorCheck((code), __FILE__, __LINE__); }

void PrintMemory(const unsigned char* memory,
                 const char label[] = "contents")
{
  std::cout << "Memory " << label << ": \n";
  for (size_t i = 0; i < 4; i++) 
  {
    for (size_t j = 0; j < 8; j++)
      printf("%02X ", static_cast<int> (memory[i * 8 + j]));
    printf("\n");
  }
}

#define M_PRINT_MEMORY(memory) \
  for (size_t i = 0; i < 7; i++)  \
  { \
    for (size_t j = 0; j < 8; j++) \
      printf("%02X ", static_cast<int> (memory[i * 8 + j])); \
    printf("\n"); \
  }

#define CPU_PRINT_MEMORY(memory, label)\
  printf("HOST   Memory "); printf(label); printf("\n"); \
  M_PRINT_MEMORY(memory) \

#define GPU_PRINT_MEMORY(memory, label)\
  forall(gpu(), 0, 1, [=] __device__ (int i) { \
    printf("DEVICE Memory "); printf(label); printf("\n"); \
    M_PRINT_MEMORY(memory) \
  });


class C
{
public:
  CHAI_HOST_DEVICE C(void) { printf("++ C has been constructed\n"); }
  CHAI_HOST_DEVICE ~C(void) { printf("-- C has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) = 0;
};

class D : public C
{
public:
  unsigned long long content_D;
  CHAI_HOST_DEVICE D(void) : content_D(0xDDDDDDDDDDDDDDDDull) { printf("++ D has been constructed\n"); }
  CHAI_HOST_DEVICE ~D(void) { printf("-- D has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) { printf("%lX\n", content_D); }
};


class A 
{
public:
  unsigned long long content_A;
  D d;
  CHAI_HOST_DEVICE A(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }
  CHAI_HOST_DEVICE ~A(void) { printf("-- A has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) = 0;
  CHAI_HOST_DEVICE virtual void d_function(void) = 0;
};

class A2
{
public:
  CHAI_HOST_DEVICE A2(void) { printf("++ A2 has been constructed\n"); }
  CHAI_HOST_DEVICE ~A2(void) { printf("-- A2 has been destructed\n"); }
};

class B : public A, public A2
{
public:
  unsigned long long content_B;
  CHAI_HOST_DEVICE B(void) : content_B(0xBBBBBBBBBBBBBBBBull) { printf("++ B has been constructed\n"); }
  CHAI_HOST_DEVICE ~B(void) { printf("-- B has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) { printf("%lX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) { d.function(); }
};



GPU_TEST(managed_ptr, polycpytest)
{

  // Assign 32 byte block of memory to 0x11 on the Host
  unsigned char* memory1 = (unsigned char*)malloc(56*sizeof(unsigned char));
  memset(memory1, 0x11, 56 * sizeof(unsigned char));
  CPU_PRINT_MEMORY(memory1, "1 : before placement new")


  // Assign 32 byte block of memory to 0x22 on the Device
  unsigned char* memory2; cudaMalloc((void**)&memory2, 56*sizeof(unsigned char));
  forall(gpu(), 0, 56, [=] __device__ (int i) {  memory2[i] = 0x22;  });
  GPU_PRINT_MEMORY(memory2, "2 : before placement new")


  // Placement New Polymorphic object on the Host.
  B* b_ptr1 = new (memory1) B;
  CPU_PRINT_MEMORY(memory1, "1 : after placement new");


  // Placement New Polymorphic object on the Device.
  B* b_ptr2 = reinterpret_cast<B*>(memory2);
  A* base2 = b_ptr2;
  forall(gpu(), 0, 1, [=] __device__ (int i) { new(b_ptr2) B();});
  GPU_PRINT_MEMORY(memory2, "2 : after placement new");


  // B was constructed on the Device so we can call virtual 
  // function on the GPU from a host pointer.
  printf("Calling virtual function from Base pointer on GPU.\n");
  forall(gpu(), 0, 1, [=] __device__ (int i) { base2->function(); });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );
  


  // Lets edit the Data on the Host...
  b_ptr1->content_B = 0xCBCBCBCBCBCBCBCBull;
  CPU_PRINT_MEMORY(memory1, "1 : after content change");
  
  // Copying Data from Host to Device
#define OFFSET_CPY
#if !defined(OFFSET_CPY)
  GPU_ERROR_CHECK(cudaMemcpy(b_ptr2, b_ptr1, sizeof(B), cudaMemcpyHostToDevice));
#else
  // We nee to skip over the Vtable and try to only copy the contents of the 
  // object itself.
  unsigned int offset = sizeof(void*);
  char* off_b_ptr2 = (char*)b_ptr2 + offset;
  char* off_b_ptr1 = (char*)b_ptr1 + offset;
  int off_size = sizeof(B) - offset;

  GPU_ERROR_CHECK(cudaMemcpy(off_b_ptr2, off_b_ptr1, off_size, cudaMemcpyHostToDevice));
  //// This will not work as we need to do pointer arithmatic at the byte level...
  //GPU_ERROR_CHECK(cudaMemcpy(b_ptr2 + offset, b_ptr1 + offset, sizeof(B) - offset, cudaMemcpyHostToDevice));
#endif
  GPU_PRINT_MEMORY(memory2, "2 : after copy from host");

  // Try to call virtual funciton on GPU like we did before.
  printf("Calling virtual function from Base pointer on GPU.\n");
  forall(gpu(), 0, 1, [=] __device__ (int i) { base2->function(); });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );



  // Lets edit the Data on the Device...
  forall(gpu(), 0, 1, [=] __device__ (int i) { 
      b_ptr2->content_B = 0xDBDBDBDBDBDBDBDBull; 
      b_ptr2->content_A = 0xDADADADADADADADAull; });
  GPU_PRINT_MEMORY(memory2, "2 : after content change");
  

#if !defined(OFFSET_CPY)
  GPU_ERROR_CHECK(cudaMemcpy(b_ptr1, b_ptr2, sizeof(B), cudaMemcpyDeviceToHost));
#else
  GPU_ERROR_CHECK(cudaMemcpy((char*)b_ptr1 + offset, (char*)b_ptr2 + offset, sizeof(B) - offset, cudaMemcpyDeviceToHost));
#endif
  CPU_PRINT_MEMORY(memory1, "1 : after copy from host");





  // Free up memory, we useed placement new so we need to call the destructor first...
  reinterpret_cast<B *>(memory1)->~B();
  forall(gpu(), 0, 1, [=] __device__ (int i) { reinterpret_cast<B*>(memory2)->~B(); });
  cudaFree(memory2);

}


