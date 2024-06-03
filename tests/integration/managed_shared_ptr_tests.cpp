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


class C : chai::CHAIPoly
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


class A : chai::CHAIPoly
{
public:
  unsigned long long content_A;
  D d;
  CHAI_HOST_DEVICE A(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }
  CHAI_HOST_DEVICE ~A(void) { printf("-- A has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const = 0;
  CHAI_HOST_DEVICE virtual void d_function(void) const = 0;
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
  CHAI_HOST_DEVICE virtual void function(void) const override { printf("%lX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) const override { d.function(); }
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long val) override { content_B = val; content_A = val; }
};


class AAbsMem : public chai::CHAICopyable , public chai::CHAIPoly
{
public:
  unsigned long long content_A;
  //chai::ManagedSharedPtr<C> base_member;
  chai::ManagedSharedPtr<const C> base_member;

  CHAI_HOST_DEVICE AAbsMem(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }

  //template<typename Derived, typename = typename std::enable_if<std::is_base_of<C, Derived>::value>::type >
  template<typename Derived>
  CHAI_HOST AAbsMem(Derived const& base_val)
    //: base_member(chai::make_shared<const Derived>(base_val))
    : base_member(chai::make_shared<Derived>(base_val))
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


GPU_TEST(managed_shared_ptr, shared_ptr_absmem)
{
  {
  using DerivedT = BAbsMem;
  using BaseT = AAbsMem;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;

  D d;
  chai::ManagedSharedPtr<BaseT> sptr = chai::make_shared<DerivedT>(d);

  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  chai::ManagedSharedPtr<const BaseT> sptr2 = sptr;
  sptr2->function();
  sptr2->d_function();

  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int i) {
    printf("CPU Body\n");
    sptr->set_content(0xFFFFFFFFFFFFFFFFull);
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
  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
}

GPU_TEST(managed_shared_ptr, shared_ptr_const)
{
  {
  using DerivedT = B;
  using BaseT = A;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;

  chai::ManagedSharedPtr<BaseT> sptr = chai::make_shared<DerivedT>();

  chai::ManagedSharedPtr<const BaseT> sptr2 = sptr;

  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int i) {
    printf("CPU Body\n");
    sptr->set_content(0xFFFFFFFFFFFFFFFFull);
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
  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
}

class NV
{
public:
  unsigned long long content_NV;
  CHAI_HOST_DEVICE NV(void) : content_NV(0xFFFFFFFFFFFFFFFFull) { printf("++ NV has been constructed\n"); }
  CHAI_HOST_DEVICE ~NV(void) { printf("-- NV has been destructed\n"); }
  CHAI_HOST_DEVICE void function(void) const { printf("%lX\n", content_NV); }
};

GPU_TEST(managed_shared_ptr, shared_ptr_nv)
{
  {
  using DerivedT = NV;
  using BaseT = A;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;

  chai::ManagedSharedPtr<DerivedT> sptr = chai::make_shared<DerivedT>();

  chai::ManagedSharedPtr<const DerivedT> sptr2 = sptr;
  //chai::ManagedSharedPtr<const BaseT> sptr2 = sptr;

  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int i) {
    printf("CPU Body\n");
    //sptr->set_content(0xFFFFFFFFFFFFFFFFull);
    sptr2->function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    sptr2->function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  }
  std::cout << "Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
}


GPU_TEST(managed_shared_ptr, shared_arr_shared_ptr_absmem)
{
  {

  //using DerivedT = NV;
  //using BaseT = NV;

  using DerivedT = BAbsMem;
  using BaseT = AAbsMem;

  using ElemT = chai::ManagedSharedPtr<BaseT>;
  using Container = chai::ManagedArray<ElemT>;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;

  std::cout << "Sptr Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;

  Container arr(1);
  D d;
  arr[0] = chai::make_shared<DerivedT>(d);
  arr.registerTouch(chai::CPU);
  //chai::ManagedSharedPtr<BaseT> sptr = chai::make_shared<DerivedT>(d);
  //arr[0] = sptr;
  ////std::cout << "Use count : " << sptr.use_count() << std::endl;
  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    arr[0]->function();
    arr[0]->d_function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int i) {
    printf("CPU Body\n");
    arr[0]->set_content(0xFFFFFFFFFFFFFFFFull);
    arr[0]->function();
    arr[0]->d_function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    printf("GPU Body\n");
    arr[0]->function();
    arr[0]->d_function();
  });
  GPU_ERROR_CHECK( cudaPeekAtLastError() );
  GPU_ERROR_CHECK( cudaDeviceSynchronize() );

  std::cout << "Sptr Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;

  arr.free();
  std::cout << "arr.free()\n";
  }
  std::cout << "Sptr Map Sz : " << chai::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;
}

