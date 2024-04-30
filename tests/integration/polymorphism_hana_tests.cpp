//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "camp/defines.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
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

#define BEGIN_EXEC_ON_DEVICE() \
  forall(gpu(), 0, 1, [=] __device__ (int i) { 

#define END_EXEC()\
  }); \
  GPU_ERROR_CHECK( cudaPeekAtLastError() );\
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );\


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








template<typename T>
Base_vtable const Base_vtable_for_host = {
   "doSomething"_s = [] __host__ __device__ (T& base){ base.doSomehting(); }
  ,"setContents"_s = [] __host__ __device__ (T& base, ull val){ base.setContents(val); }
};

template<typename T>
Base_vtable const Base_vtable_for_host = {
   [] __host__ __device__ (void* this_){ static_cast<T*>(this_)->doSomething(); }
  ,[] __host__ __device__ (void* this_, unsigned long long val){ static_cast<T*>(this_)->setContents(val); }
};

template<typename T>
__global__
void Base_vtable_for_device(Base_vtable* vptr_) {
  new(vptr_) Base_vtable{[] __host__ __device__ (void* this_){ static_cast<T*>(this_)->doSomething(); }
                        ,[] __host__ __device__ (void* this_, unsigned long long val){ static_cast<T*>(this_)->setContents(val); }
  };
};



//-----------------------------------------------------------------------------

template<typename T>
Base_vtable* make_Base_vtable_on_device() {
  Base_vtable* vptr_;
  cudaMalloc((void**)&vptr_, sizeof(Base_vtable));
  Base_vtable_for_device<T> <<<1,1>>>(vptr_);
  return vptr_;
}

struct CHAIPolyInterface {

  template <typename Any>
  CHAIPolyInterface(Any base)
  { 
    vtbl_host_ = &Base_vtable_for_host<Any>;
    ptr_host_ = new Any{base}; 

    vtbl_device_ = make_Base_vtable_on_device<Any>();
    cudaMalloc(&ptr_device_, sizeof(Any));

    obj_size_ = sizeof(Any);
  }

  void move(chai::ExecutionSpace space)
  {
    if (space == chai::CPU) cudaMemcpy(ptr_host_, ptr_device_, obj_size_, cudaMemcpyDeviceToHost);
    if (space == chai::GPU) cudaMemcpy(ptr_device_, ptr_host_, obj_size_, cudaMemcpyHostToDevice);
  }

protected:
  Base_vtable const* vtbl_host_;
  Base_vtable* vtbl_device_;
  //void* ptr_; 
  void* ptr_host_;
  void* ptr_device_;

  long obj_size_;

};

//-----------------------------------------------------------------------------

#include<iostream>
#include<vector>

struct IBase : decltype(camp::requires(
      "doSomething"_s = camp::function<void(camp::T&)>
      "setContents"_s = camp::function<void(camp::T&, ull val)>
)) {};

struct Base {
  template <typename Any>
  Base(Any base) : poly_(base) {};

  CHAI_HOST_DEVICE void doSomething() const { poly_.virtual("doSomething"_s)(poly_); }
  CHAI_HOST_DEVICE void setContents(unsigned long long val) const { poly.virtual("setContents")(poly_, val); }

private:
  ChaiPolyInterface<IBase> poly_;
};


struct DerivedA {
  CHAI_HOST_DEVICE void doSomething() { printf("DerivedA: doSomething\n"); }
  CHAI_HOST_DEVICE void setContents(unsigned long long) {}
};

struct DerivedB {
  DerivedB() : content(0xDDDDDDDDDDDDDDDDull) {};

  void doBthing() { printf("concrete B thing"); }

  CHAI_HOST_DEVICE void doSomething() { printf("DerivedB: doSomething : %lX\n", content); }
  CHAI_HOST_DEVICE void setContents(unsigned long long val) { content = val; }

private:
  unsigned long long content;
};


GPU_TEST(managed_ptr, customvtabletest) {

  Base b = Base(DerivedA{});
  Base b2 = Base(DerivedB{});

  b.doSomething();
  b2.doSomething();
  
  b.move(chai::GPU);
  b2.move(chai::GPU);

  BEGIN_EXEC_ON_DEVICE()
    printf("-- GPU Kernel begin\n");
    b.doSomething();
    b2.doSomething();
    printf("-- GPU Kernel end\n");
  END_EXEC()


  b2.setContents(0xCCCCCCCCCCCCCCCCull);
  b2.move(chai::GPU);

  BEGIN_EXEC_ON_DEVICE()
    printf("-- GPU Kernel begin\n");
    b.doSomething();
    b2.doSomething();
    b2.setContents(0xBBBBBBBBBBBBBBBBull);
    printf("-- GPU Kernel end\n");
  END_EXEC()

  b2.move(chai::CPU);
  b2.doSomething();





}

