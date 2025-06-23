//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "chai/config.hpp"
#include "chai/expt/HostManager.hpp"
#include "umpire/ResourceManager.hpp"

class HostManagerTest : public testing::Test {
  protected:
    int m_allocator_id{umpire::ResourceManager::getInstance().getAllocator("HOST").getId()};
    std::size_t m_size{100};
    chai::expt::HostManager m_manager{m_allocator_id, m_size};
};


TEST_F(HostManagerTest, AllocatorID)
{
  EXPECT_EQ(m_allocator_id, m_manager.getAllocatorID());
}

TEST_F(HostManagerTest, Size)
{
  EXPECT_EQ(m_size, m_manager.size());
}

TEST_F(HostManagerTest, DataExecutionContextNone)
{
  EXPECT_EQ(nullptr, m_manager.data(chai::expt::ExecutionContext::NONE, false));
}

TEST_F(HostManagerTest, DataExecutionContextHost)
{
  EXPECT_NE(nullptr, m_manager.data(chai::expt::ExecutionContext::HOST, false));
}

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
TEST_F(HostManagerTest, DataExecutionContextDevice)
{
  EXPECT_EQ(nullptr, m_manager.data(chai::expt::ExecutionContext::DEVICE, false));
}
#endif

TEST(PinnedManagerTest, Container1)
{
   // Could have accessor parameter to control whether operator[] is defined.
   chai::ManagedArray<int> myArray(10);

   {
      ManagedView<int> myView(myArray);

      RAJA_LOOP(i, 0, 10) {
         // Could accidentally do a deep copy of myArray if didn't cast to a view
         myView[i]++;
      } RAJA_LOOP_END
   }

   ManagedArray<int> myArray2 = myArray; // Deep copy
}

TEST(PinnedManagerTest, Container2)
{
   ManagedArray<int> myArray(10);

   RAJA_LOOP(i, 0, 10, myView = ManagedView<int>(myArray)) {
      // Could accidentally do a deep copy of myArray if didn't cast to a view
      myView[i]++;
   } RAJA_LOOP_END

   ManagedArray<int> myArray2 = myArray; // Deep copy
}


TEST(PinnedManagerTest, UniquePtr1)
{
   ManagedArray<int> myArray(10);

   {
      ManagedView<int> myView(myArray);

      // Can't accidentally copy myArray into loop
      RAJA_LOOP(i, 0, 10) {
         myView[i]++;
      } RAJA_LOOP_END
   }

   ManagedArray<int>& myArray2 = myArray; // Can't do a deep copy
}

TEST(PinnedManagerTest, UniqueArray)
{
   chai::UniqueArray<int> myArray = chai::makeUnique(10);

   // Can't accidentally copy myArray into loop
   RAJA_LOOP(i, 0, 10, myView = ManagedView<int>(myArray)) {
      myView[i]++;
   } RAJA_LOOP_END

   // Can't do a deep copy
   ManagedArray<int> myArray2 = myArray; // Deep copy
}

TEST(PinnedManagerTest, SharedArray)
{
   chai::SharedArray<int> myArray = chai::makeShared(10);

   RAJA_LOOP(i, 0, 10) {
      // Can be used directly in RAJA loop
      myArray[i]++;
   } RAJA_LOOP_END
}

TEST(PinnedManagerTest, NonOwnedArray)
{
   chai::NonOwnedArray<int> myArray = chai::makeNonOwned(10);

   RAJA_LOOP(i, 0, 10) {
      // Can be used directly in RAJA loop
      myArray[i]++;
   } RAJA_LOOP_END

   // Have to explicitly manage lifetime
   myArray.free();
}
