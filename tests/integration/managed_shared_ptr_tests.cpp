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
#include "chai/DeviceHelpers.hpp"

#include "../src/util/forall.hpp"

// Standard library headers
#include <cstdlib>

#ifdef CHAI_DISABLE_RM
#define assert_empty_array_map(IGNORED)
#define assert_empty_sptr_map(IGNORED)
#else
#define assert_empty_array_map(IGNORED) ASSERT_EQ(chai::ArrayManager::getInstance()->getPointerMap().size(),0)
#define assert_empty_sptr_map(IGNORED) ASSERT_EQ(chai::expt::SharedPtrManager::getInstance()->getPointerMap().size(),0)
#endif


class C : chai::expt::CHAIPoly
{
public:
  CHAI_HOST_DEVICE C(void) { printf("++ C has been constructed\n"); }
  CHAI_HOST_DEVICE virtual ~C(void) { printf("-- C has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const = 0;
};

class D final : public C
{
public:
  unsigned long long content_D;
  CHAI_HOST_DEVICE D(void) : content_D(0xDDDDDDDDDDDDDDDDull) { printf("++ D has been constructed\n"); }
  CHAI_HOST_DEVICE ~D(void) { printf("-- D has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const { printf("%llX\n", content_D); }
};


class A : chai::expt::CHAIPoly
{
public:
  unsigned long long content_A;
  D d;
  CHAI_HOST_DEVICE A(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }
  CHAI_HOST_DEVICE virtual ~A(void) { printf("-- A has been destructed\n"); }
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

class B final : public A, public A2
{
public:
  unsigned long long content_B;
  CHAI_HOST_DEVICE B(void) : content_B(0xBBBBBBBBBBBBBBBBull) { printf("++ B has been constructed\n"); }
  CHAI_HOST_DEVICE ~B(void) { printf("-- B has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const override { printf("%llX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) const override { d.function(); }
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long val) override { content_B = val; content_A = val; }
};


class AAbsMem : public chai::CHAICopyable , public chai::expt::CHAIPoly
{
public:
  chai::expt::ManagedSharedPtr<const C> base_member;
  unsigned long long content_A;

  CHAI_HOST_DEVICE AAbsMem(void) : content_A(0xAAAAAAAAAAAAAAAAull) { printf("++ A has been constructed\n"); }

  template<typename Derived>
  CHAI_HOST AAbsMem(Derived const& base_val)
    : base_member(chai::expt::make_shared<Derived>(base_val))
    , content_A(0xAAAAAAAAAAAAAAAAull) 
  { printf("++ A has been constructed\n"); }

  CHAI_HOST_DEVICE virtual ~AAbsMem(void) { printf("-- A has been destructed\n"); }
  CHAI_HOST_DEVICE virtual void function(void) const = 0;
  CHAI_HOST_DEVICE virtual void d_function(void) const = 0;
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long) = 0;
};

class BAbsMem final : public AAbsMem
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
  CHAI_HOST_DEVICE virtual void function(void) const override { printf("%llX\n", content_B); }
  CHAI_HOST_DEVICE virtual void d_function(void) const override { base_member->function(); }
  CHAI_HOST_DEVICE virtual void set_content(unsigned long long val) override { content_B = val; content_A = val; }
};

class NV
{
public:
  unsigned long long content_NV;
  CHAI_HOST_DEVICE NV(void) : content_NV(0xFFFFFFFFFFFFFFFFull) { printf("++ NV has been constructed\n"); }
  CHAI_HOST_DEVICE ~NV(void) { printf("-- NV has been destructed\n"); }
  CHAI_HOST_DEVICE void function(void) const { printf("%llX\n", content_NV); }
};

/*
 * This test ensures that dynamic dispatch works for types that are polymorphic
 * who themselves own a polymorphic member.
 *
 * - function: exercises dynamic dispatch directly for the Derived type in the
 *   shared ptr.
 * - d_function: calls a second level of dynamic dispatch on an owned
 *   ManagedSharedPtr member.
 */
GPU_TEST(managed_shared_ptr, shared_ptr_absmem)
{
  {
  using DerivedT = BAbsMem;
  using BaseT = AAbsMem;

  D d;
  chai::expt::ManagedSharedPtr<BaseT> sptr = chai::expt::make_shared<DerivedT>(d);

  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  chai::expt::ManagedSharedPtr<const BaseT> sptr2 = sptr;
  sptr2->function();
  sptr2->d_function();

  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int) {
    printf("CPU Body\n");
    sptr->set_content(0xFFFFFFFFFFFFFFFFull);
    sptr2->function();
    sptr2->d_function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  }
  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  assert_empty_sptr_map();
}

/*
 * This demonstrates casting / copying ManagedSharedPtr types to const T. We
 * then call const and non const methods from their respective variable.
 */
GPU_TEST(managed_shared_ptr, shared_ptr_const)
{
  {
  using DerivedT = B;
  using BaseT = A;

  std::cout << "size of (DerivedT) : " << sizeof(DerivedT) << std::endl;
  std::cout << "size of (BaseT)    : " << sizeof(BaseT)    << std::endl;

  chai::expt::ManagedSharedPtr<BaseT> sptr = chai::expt::make_shared<DerivedT>();

  chai::expt::ManagedSharedPtr<const BaseT> sptr2 = sptr;

  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int) {
    printf("CPU Body\n");
    sptr->set_content(0xFFFFFFFFFFFFFFFFull);
    sptr2->function();
    sptr2->d_function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
    sptr2->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  }
  assert_empty_sptr_map();
  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
}

/*
 * This test ensures ManagedSharedPtr will move a type without a virtual table
 * correctly between the host and the device.
 */
GPU_TEST(managed_shared_ptr, shared_ptr_nv)
{
  {
  using DerivedT = NV;

  chai::expt::ManagedSharedPtr<DerivedT> sptr = chai::expt::make_shared<DerivedT>();

  chai::expt::ManagedSharedPtr<const DerivedT> sptr2 = sptr;

  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int) {
    printf("CPU Body\n");
    sptr2->function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    sptr2->function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  }
  assert_empty_sptr_map();
  std::cout << "Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
}

/*
 * This test checks that ManagedArrays of ManagedSharedPtr types
 * correctly copy objects between the host and the device.
 */
GPU_TEST(managed_shared_ptr, shared_arr_shared_ptr_absmem)
{
  {
  using DerivedT = BAbsMem;
  using BaseT = AAbsMem;

  using ElemT = chai::expt::ManagedSharedPtr<BaseT>;
  using Container = chai::ManagedArray<ElemT>;

  std::cout << "Sptr Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;

  Container arr(1);
  D d;
  arr[0] = chai::expt::make_shared<DerivedT>(d);
  arr.registerTouch(chai::CPU);

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    arr[0]->function();
    arr[0]->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  std::cout << "CPU CALL...\n";
  forall(sequential(), 0, 1, [=] (int) {
    printf("CPU Body\n");
    arr[0]->set_content(0xFFFFFFFFFFFFFFFFull);
    arr[0]->function();
    arr[0]->d_function();
  });

  std::cout << "GPU CALL...\n";
  forall(gpu(), 0, 1, [=] __device__ (int) {
    printf("GPU Body\n");
    arr[0]->function();
    arr[0]->d_function();
  });
  CHAI_GPU_ERROR_CHECK( gpuPeekAtLastError() );
  CHAI_GPU_ERROR_CHECK( gpuDeviceSynchronize() );

  std::cout << "Sptr Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;

  std::cout << "arr.free()\n";
  arr.free();
  std::cout << "End of scope\n";
  assert_empty_array_map();

  }
  std::cout << "Sptr Map Sz : " << chai::expt::SharedPtrManager::getInstance()->getPointerMap().size() << std::endl;
  std::cout << "Arr  Map Sz : " << chai::ArrayManager::getInstance()->getPointerMap().size() << std::endl;
  assert_empty_sptr_map();
}

