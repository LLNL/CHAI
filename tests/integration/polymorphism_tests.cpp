//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <type_traits>
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


class C
{
public:
  CHAI_HOST_DEVICE C(void) { printf("++ C has been constructed\n"); }
  CHAI_HOST_DEVICE ~C(void) { printf("-- C has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const = 0;
};

class D : public C
{
public:
  unsigned long long content_D;
  CHAI_HOST_DEVICE D(void) : content_D(0xDDDDDDDDDDDDDDDDull) { printf("++ D has been constructed\n"); }
  CHAI_HOST_DEVICE ~D(void) { printf("-- D has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const { printf("%lX\n", content_D); }
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
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long) = 0;
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
  CHAI_HOST_DEVICE virtual void function(void) override { printf("%lX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) override { d.function(); }
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long val) override { content_B = val; content_A = val; }
};


class AAbsMem : public chai::CHAICopyable 
{
public:
  unsigned long long content_A;
  chai::ManagedSharedPtr<const C> base_member;

  CHAI_HOST_DEVICE AAbsMem(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }

  //template<typename Derived, typename = typename std::enable_if<std::is_base_of<C, Derived>::value>::type >
  template<typename Derived>
  CHAI_HOST AAbsMem(Derived const& base_val)
    : base_member(chai::make_shared<const Derived>(base_val))
    , content_A(0xAAAAAAAAAAAAAAAAull) 
  { printf("++ A has been constructed\n"); }

  CHAI_HOST_DEVICE ~AAbsMem(void) { printf("-- A has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const = 0;
  CHAI_HOST_DEVICE virtual void d_function(void) const = 0;
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long) = 0;
};

class BAbsMem : public AAbsMem
{
public:
  unsigned long long content_B;

  CHAI_HOST_DEVICE BAbsMem() : AAbsMem()
  { 
    printf("++ B has been constructed\n"); 
  }

  template<typename Derived>
  CHAI_HOST BAbsMem(Derived const& base_val) 
    : AAbsMem(base_val)
    , content_B(0xBBBBBBBBBBBBBBBBull) 
  { 
    printf("++ B has been constructed\n"); 
  }

  CHAI_HOST_DEVICE ~BAbsMem(void) { printf("-- B has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const override { printf("%lX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) const override { base_member->function(); }
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long val) override { content_B = val; content_A = val; }
};


GPU_TEST(managed_ptr, shared_ptr_absmem)
{
  {
  using DerivedT = BAbsMem;
  using BaseT = AAbsMem;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;

  //chai::ManagedSharedPtr<DerivedT> sptr = chai::make_shared<DerivedT>(D{});
  D d;
  //DerivedT der(d);
  chai::ManagedSharedPtr<const BaseT> sptr = chai::make_shared<const DerivedT>(d);
  //chai::ManagedSharedPtr<BaseT> sptr = chai::make_shared<DerivedT>(d);

  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  chai::ManagedSharedPtr<const BaseT> sptr2 = sptr;
  sptr2->function();
  sptr2->d_function();

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });

  //sptr2.registerTouch(chai::GPU);


  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int i) {
    printf("CPU Body\n");
    //sptr->set_content(0xFFFFFFFFFFFFFFFFull);
    sptr2->function();
    sptr2->d_function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  }
}

//GPU_TEST(managed_ptr, shared_ptr)
//{
//  {
//  using DerivedT = B;
//  using BaseT = A;
//
//  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
//  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;
//
//  chai::ManagedSharedPtr<DerivedT> sptr = chai::make_shared_deleter<DerivedT>(
//      [](DerivedT* p){ printf("Custom Deleter Call\n"); p->~DerivedT(); });
//
//  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
//
//  chai::ManagedSharedPtr<BaseT> sptr2 = sptr;
//  std::cout << "use_count : " << sptr.use_count() << std::endl;
//
//  sptr->set_content(0xFFFFFFFFFFFFFFFFull);
//
//  std::cout << "GPU CALL...\n";
//  forall(gpu(), 0, 1, [=] __device__ (int i) {
//    printf("GPU Body\n");
//    sptr2->function();
//    sptr2->d_function();
//
//    //results[i] = rawArrayClass->getValue(i);
//  });
//  GPU_ERROR_CHECK( cudaPeekAtLastError() );
//  GPU_ERROR_CHECK( cudaDeviceSynchronize() );
//
//  }
//}
//
//
//
//GPU_TEST(managed_ptr, shared_ptralloc)
//{
//
//  {
//
//  using DerivedT = B;
//  using BaseT = A;
//
//  chai::SharedPtrManager* sptr_manager = chai::SharedPtrManager::getInstance();
//  umpire::ResourceManager& res_manager = umpire::ResourceManager::getInstance();
//
//  auto cpu_allocator = sptr_manager->getAllocator(chai::CPU);
//  BaseT* cpu_ptr = static_cast<DerivedT*>( cpu_allocator.allocate(1*sizeof(DerivedT)) );
//
//  new(cpu_ptr) DerivedT();
//  BaseT* gpu_ptr = chai::msp_make_on_device<DerivedT>();
//
//
//  auto record = sptr_manager->makeSharedPtrRecord(cpu_ptr, gpu_ptr, sizeof(DerivedT), true);
//
//  forall(gpu(), 0, 1, [=] __device__ (int i) {
//    printf("GPU Body\n");
//    gpu_ptr->function();
//    gpu_ptr->d_function();
//  });
//
//  std::cout << "Ump alloc cpu : " << cpu_ptr << std::endl;
//  std::cout << "Ump alloc gpu : " << gpu_ptr << std::endl;
//  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
//
//  cpu_ptr->set_content(0xFFFFFFFFFFFFFFFFull);
//
//  camp::resources::Resource device_resource(camp::resources::Cuda::get_default());
//  res_manager.copy_poly(gpu_ptr, cpu_ptr, device_resource);
//
//  //unsigned int offset = sizeof(void*);
//  //GPU_ERROR_CHECK(cudaMemcpy((char*)gpu_ptr+offset, (char*)cpu_ptr+offset, sizeof(DerivedT)-offset, cudaMemcpyHostToDevice));
//  //GPU_ERROR_CHECK(cudaMemcpy((char*)gpu_ptr, (char*)cpu_ptr, sizeof(DerivedT), cudaMemcpyHostToDevice));
//
//  forall(gpu(), 0, 1, [=] __device__ (int i) {
//    printf("GPU Body\n");
//    gpu_ptr->function();
//    gpu_ptr->d_function();
//  });
//  GPU_ERROR_CHECK( cudaPeekAtLastError() );
//  GPU_ERROR_CHECK( cudaDeviceSynchronize() );
//
//  }
//  //assert_empty_map();
//}
//
//GPU_TEST(managed_ptr, polycpytest)
//{
//
//  // Assign 32 byte block of memory to 0x11 on the Host
//  unsigned char* memory1 = (unsigned char*)malloc(56*sizeof(unsigned char));
//  memset(memory1, 0x11, 56 * sizeof(unsigned char));
//  CPU_PRINT_MEMORY(memory1, "1 : before placement new")
//
//
//  // Assign 32 byte block of memory to 0x22 on the Device
//  unsigned char* memory2; cudaMalloc((void**)&memory2, 56*sizeof(unsigned char));
//  forall(gpu(), 0, 56, [=] __device__ (int i) {  memory2[i] = 0x22;  });
//  GPU_PRINT_MEMORY(memory2, "2 : before placement new")
//
//
//  // Placement New Polymorphic object on the Host.
//  B* b_ptr1 = new (memory1) B;
//  CPU_PRINT_MEMORY(memory1, "1 : after placement new");
//
//
//  // Placement New Polymorphic object on the Device.
//  B* b_ptr2 = reinterpret_cast<B*>(memory2);
//  A* base2 = b_ptr2;
//  forall(gpu(), 0, 1, [=] __device__ (int i) { new(b_ptr2) B();});
//  GPU_PRINT_MEMORY(memory2, "2 : after placement new");
//
//
//  // B was constructed on the Device so we can call virtual 
//  // function on the GPU from a host pointer.
//  printf("Calling virtual function from Base pointer on GPU.\n");
//  forall(gpu(), 0, 1, [=] __device__ (int i) { base2->function(); });
//  GPU_ERROR_CHECK( cudaPeekAtLastError() );
//  GPU_ERROR_CHECK( cudaDeviceSynchronize() );
//  
//
//
//  // Lets edit the Data on the Host...
//  b_ptr1->content_B = 0xCBCBCBCBCBCBCBCBull;
//  CPU_PRINT_MEMORY(memory1, "1 : after content change");
//  
//  // Copying Data from Host to Device
//#define OFFSET_CPY
//#if !defined(OFFSET_CPY)
//  GPU_ERROR_CHECK(cudaMemcpy(b_ptr2, b_ptr1, sizeof(B), cudaMemcpyHostToDevice));
//#else
//  // We nee to skip over the Vtable and try to only copy the contents of the 
//  // object itself.
//  unsigned int offset = sizeof(void*);
//  char* off_b_ptr2 = (char*)b_ptr2 + offset;
//  char* off_b_ptr1 = (char*)b_ptr1 + offset;
//  int off_size = sizeof(B) - offset;
//
//  GPU_ERROR_CHECK(cudaMemcpy(off_b_ptr2, off_b_ptr1, off_size, cudaMemcpyHostToDevice));
//  //// This will not work as we need to do pointer arithmatic at the byte level...
//  //GPU_ERROR_CHECK(cudaMemcpy(b_ptr2 + offset, b_ptr1 + offset, sizeof(B) - offset, cudaMemcpyHostToDevice));
//#endif
//  GPU_PRINT_MEMORY(memory2, "2 : after copy from host");
//
//  // Try to call virtual funciton on GPU like we did before.
//  printf("Calling virtual function from Base pointer on GPU.\n");
//  forall(gpu(), 0, 1, [=] __device__ (int i) { base2->function(); });
//  GPU_ERROR_CHECK( cudaPeekAtLastError() );
//  GPU_ERROR_CHECK( cudaDeviceSynchronize() );
//
//
//
//  // Lets edit the Data on the Device...
//  forall(gpu(), 0, 1, [=] __device__ (int i) { 
//      b_ptr2->content_B = 0xDBDBDBDBDBDBDBDBull; 
//      b_ptr2->content_A = 0xDADADADADADADADAull; });
//  GPU_PRINT_MEMORY(memory2, "2 : after content change");
//  
//
//#if !defined(OFFSET_CPY)
//  GPU_ERROR_CHECK(cudaMemcpy(b_ptr1, b_ptr2, sizeof(B), cudaMemcpyDeviceToHost));
//#else
//  GPU_ERROR_CHECK(cudaMemcpy((char*)b_ptr1 + offset, (char*)b_ptr2 + offset, sizeof(B) - offset, cudaMemcpyDeviceToHost));
//#endif
//  CPU_PRINT_MEMORY(memory1, "1 : after copy from host");
//
//  // Free up memory, we useed placement new so we need to call the destructor first...
//  reinterpret_cast<B *>(memory1)->~B();
//  forall(gpu(), 0, 1, [=] __device__ (int i) { reinterpret_cast<B*>(memory2)->~B(); });
//  cudaFree(memory2);
//
//}
//
//
//
//
//
//
//
//
//
//struct Base_vtable {
//  void (*doSomething)(void* this_);
//  void (*setContents)(void* this_, unsigned long long val);
//};
//
//template<typename T>
//Base_vtable const Base_vtable_for_host = {
//   [] __host__ __device__ (void* this_){ static_cast<T*>(this_)->doSomething(); }
//  ,[] __host__ __device__ (void* this_, unsigned long long val){ static_cast<T*>(this_)->setContents(val); }
//};
//
//template<typename T>
//__global__
//void Base_vtable_for_device(Base_vtable* vptr_) {
//  new(vptr_) Base_vtable{[] __host__ __device__ (void* this_){ static_cast<T*>(this_)->doSomething(); }
//                        ,[] __host__ __device__ (void* this_, unsigned long long val){ static_cast<T*>(this_)->setContents(val); }
//  };
//};
//
//
//
////-----------------------------------------------------------------------------
//
//#if !defined(CHAI_DEVICE_COMPILE)
//#define CHAI_POLY_VIRTUAL_CALL(name) \
//    return vtbl_host_->name((void*) ptr_host_); 
//#define CHAI_POLY_VIRTUAL_CALL_ARGS(name, ...) \
//    return vtbl_host_->name((void*) ptr_host_, __VA_ARGS__); 
//#else
//#define CHAI_POLY_VIRTUAL_CALL(name) \
//    return vtbl_device_->name((void*) ptr_device_);
//#define CHAI_POLY_VIRTUAL_CALL_ARGS(name, ...) \
//    return vtbl_device_->name((void*) ptr_device_, __VA_ARGS__);
//#endif
//
//template<typename T>
//Base_vtable* make_Base_vtable_on_device() {
//  Base_vtable* vptr_;
//  cudaMalloc((void**)&vptr_, sizeof(Base_vtable));
//  Base_vtable_for_device<T> <<<1,1>>>(vptr_);
//  return vptr_;
//}
//
//struct CHAIPolyInterface {
//
//  template <typename Any>
//  CHAIPolyInterface(Any base)
//  { 
//    vtbl_host_ = &Base_vtable_for_host<Any>;
//    ptr_host_ = new Any{base}; 
//
//    vtbl_device_ = make_Base_vtable_on_device<Any>();
//    cudaMalloc(&ptr_device_, sizeof(Any));
//
//    obj_size_ = sizeof(Any);
//  }
//
//  void move(chai::ExecutionSpace space)
//  {
//    if (space == chai::CPU) cudaMemcpy(ptr_host_, ptr_device_, obj_size_, cudaMemcpyDeviceToHost);
//    if (space == chai::GPU) cudaMemcpy(ptr_device_, ptr_host_, obj_size_, cudaMemcpyHostToDevice);
//  }
//
//protected:
//  Base_vtable const* vtbl_host_;
//  Base_vtable* vtbl_device_;
//  //void* ptr_; 
//  void* ptr_host_;
//  void* ptr_device_;
//
//  long obj_size_;
//
//};
//
////-----------------------------------------------------------------------------
//
//#include<iostream>
//#include<vector>
//
//
//struct Base: CHAIPolyInterface {
//  using Poly = CHAIPolyInterface;
//
//  template <typename Any>
//  Base(Any base) : Poly(base) {};
//
//  CHAI_HOST_DEVICE void doSomething() const { CHAI_POLY_VIRTUAL_CALL(doSomething) }
//  CHAI_HOST_DEVICE void setContents(unsigned long long val) const { CHAI_POLY_VIRTUAL_CALL_ARGS(setContents, val) }
//};
//
//
//struct DerivedA {
//  CHAI_HOST_DEVICE void doSomething() { printf("DerivedA: doSomething\n"); }
//  CHAI_HOST_DEVICE void setContents(unsigned long long) {}
//};
//
//struct DerivedB {
//  DerivedB() : content(0xDDDDDDDDDDDDDDDDull) {};
//
//  void doBthing() { printf("concrete B thing"); }
//
//  CHAI_HOST_DEVICE void doSomething()
//  { 
//    printf("DerivedB: doSomething\n"); 
//    printf("%lX\n", content);
//  }
//  CHAI_HOST_DEVICE void setContents(unsigned long long val) { content = val; }
//  unsigned long long content;
//};
//
//
//GPU_TEST(managed_ptr, customvtabletest) {
//
//  Base b = Base(DerivedA{});
//  Base b2 = Base(DerivedB{});
//
//  b.doSomething();
//  b2.doSomething();
//  
//  b.move(chai::GPU);
//  b2.move(chai::GPU);
//
//  BEGIN_EXEC_ON_DEVICE()
//    printf("-- GPU Kernel begin\n");
//    b.doSomething();
//    b2.doSomething();
//    printf("-- GPU Kernel end\n");
//  END_EXEC()
//
//
//  b2.setContents(0xCCCCCCCCCCCCCCCCull);
//  b2.move(chai::GPU);
//
//  BEGIN_EXEC_ON_DEVICE()
//    printf("-- GPU Kernel begin\n");
//    b.doSomething();
//    b2.doSomething();
//    b2.setContents(0xBBBBBBBBBBBBBBBBull);
//    printf("-- GPU Kernel end\n");
//  END_EXEC()
//
//  b2.move(chai::CPU);
//  b2.doSomething();
//
//
//
//
//
//}

