//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/PointerRecord.hpp"

TEST(ArrayManager, Constructor)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  ASSERT_NE(rm, nullptr);
}

#ifndef CHAI_DISABLE_RM

TEST(ArrayManager, getPointerMap)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  // Allocate one array
  size_t sizeOfArray1 = 5;
  chai::ManagedArray<int> array1 =
      chai::ManagedArray<int>(sizeOfArray1, chai::CPU);

  // Check map of pointers
  std::unordered_map<void*, const chai::PointerRecord*> map1 =
      rm->getPointerMap();
  ASSERT_EQ(map1.size(), 1);

  // Check some of the entries in the pointer record
  ASSERT_TRUE(map1.find(array1.data()) != map1.end());
  const chai::PointerRecord* record1Temp = map1[array1.data()];
  ASSERT_EQ(record1Temp->m_size, sizeOfArray1 * sizeof(int));
  ASSERT_EQ(record1Temp->m_last_space, chai::CPU);

  // Check total num arrays and total allocated memory
  ASSERT_EQ(rm->getTotalNumArrays(), 1);
  ASSERT_EQ(rm->getTotalSize(), sizeOfArray1 * sizeof(int));

  // Allocate another array
  size_t sizeOfArray2 = 4;
  chai::ManagedArray<double> array2 =
      chai::ManagedArray<double>(sizeOfArray2, chai::CPU);

  // Check map of pointers
  std::unordered_map<void*, const chai::PointerRecord*> map2 =
      rm->getPointerMap();
  ASSERT_EQ(map2.size(), 2);

  // Check that the entries in the first record are not changed
  ASSERT_TRUE(map2.find(array1.data()) != map2.end());
  const chai::PointerRecord* record1 = map1[array1.data()];
  ASSERT_EQ(record1->m_size, sizeOfArray1 * sizeof(int));
  ASSERT_EQ(record1->m_last_space, chai::CPU);

  // Check some of the entries in the pointer record
  ASSERT_TRUE(map2.find(array2.data()) != map2.end());
  const chai::PointerRecord* record2 = map2[array2.data()];
  ASSERT_EQ(record2->m_size, sizeOfArray2 * sizeof(double));
  ASSERT_EQ(record2->m_last_space, chai::CPU);

  // Check the equality of the records
  ASSERT_EQ(record1, record1Temp);
  ASSERT_NE(record1, record2);

  // Check total num arrays and total allocated memory
  ASSERT_EQ(rm->getTotalNumArrays(), 2);
  ASSERT_EQ(rm->getTotalSize(),
            (sizeOfArray1 * sizeof(int)) + (sizeOfArray2 * sizeof(double)));

  array1.free();
  array2.free();
}

/*!
 * \brief Tests to see if callbacks can be turned on or off
 */
TEST(ArrayManager, controlCallbacks)
{
  // First check that callbacks are turned on by default
  chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();

  // Variable for testing if callbacks are on or off
  bool callbacksAreOn = false;

  // Allocate one array and set a callback
  size_t sizeOfArray = 5;
  chai::ManagedArray<int> array1(sizeOfArray, chai::CPU);
  array1.setUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                           callbacksAreOn = true;
                         });

  // Make sure the callback is called with ACTION_FREE
  array1.free();
  ASSERT_TRUE(callbacksAreOn);

  // Now turn off callbacks
  arrayManager->disableCallbacks();

  // Reset the variable for testing if callbacks are on or off
  callbacksAreOn = false;

  // Allocate another array and set a callback
  chai::ManagedArray<int> array2(sizeOfArray, chai::CPU);
  array2.setUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                           callbacksAreOn = true;
                         });

  // Make sure the callback is called with ACTION_FREE
  array2.free();
  ASSERT_FALSE(callbacksAreOn);

  // Now make sure the order doesn't matter for when the callback is set compared
  // to when callbacks are enabled

  // Reset the variable for testing if callbacks are on or off
  callbacksAreOn = false;

  // Allocate a third array and set a callback
  chai::ManagedArray<int> array3(sizeOfArray, chai::CPU);
  array3.setUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                           callbacksAreOn = true;
                         });

  // Turn on callbacks
  arrayManager->enableCallbacks();

  // Make sure the callback is called with ACTION_FREE
  array3.free();
  ASSERT_TRUE(callbacksAreOn);
}

/*!
 * \brief Tests to see if global callback can be turned on or off
 */
TEST(ArrayManager, controlGlobalCallback)
{
  // First check that callbacks are turned on by default
  chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();

  // Variable for testing if callbacks are on or off
  bool callbacksAreOn = false;

  // Set a global callback
  arrayManager->setGlobalUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                                        callbacksAreOn = true;
                                      });

  // Allocate an array and make sure the callback was called
  size_t sizeOfArray = 5;
  chai::ManagedArray<int> array(sizeOfArray, chai::CPU);
  ASSERT_TRUE(callbacksAreOn);

  // Now turn off callbacks
  arrayManager->disableCallbacks();

  // Reset the variable for testing if callbacks are on or off
  callbacksAreOn = false;

  // Realloc the array and make sure the callback was NOT called
  array.reallocate(2 * sizeOfArray);
  ASSERT_FALSE(callbacksAreOn);

  // Now make sure the order doesn't matter for when the callback is set compared
  // to when callbacks are enabled
  arrayManager->setGlobalUserCallback([&] (const chai::PointerRecord*, chai::Action, chai::ExecutionSpace) {
                                        callbacksAreOn = true;
                                      });

  // Reset the variable for testing if callbacks are on or off
  callbacksAreOn = false;

  // Turn on callbacks
  arrayManager->enableCallbacks();

  // Make sure the callback is called
  array.free();
  ASSERT_TRUE(callbacksAreOn);
}

#endif // !CHAI_DISABLE_RM
