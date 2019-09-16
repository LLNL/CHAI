// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
// rights reserved.
//
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of CHAI.
//
// LLNL-CODE-705877
//
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
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
  ASSERT_TRUE(map1.find(array1) != map1.end());
  const chai::PointerRecord* record1Temp = map1[array1];
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
  ASSERT_TRUE(map2.find(array1) != map2.end());
  const chai::PointerRecord* record1 = map1[array1];
  ASSERT_EQ(record1->m_size, sizeOfArray1 * sizeof(int));
  ASSERT_EQ(record1->m_last_space, chai::CPU);

  // Check some of the entries in the pointer record
  ASSERT_TRUE(map2.find(array2) != map2.end());
  const chai::PointerRecord* record2 = map2[array2];
  ASSERT_EQ(record2->m_size, sizeOfArray2 * sizeof(double));
  ASSERT_EQ(record2->m_last_space, chai::CPU);

  // Check the equality of the records
  ASSERT_EQ(record1, record1Temp);
  ASSERT_NE(record1, record2);

  // Check total num arrays and total allocated memory
  ASSERT_EQ(rm->getTotalNumArrays(), 2);
  ASSERT_EQ(rm->getTotalSize(),
            (sizeOfArray1 * sizeof(int)) + (sizeOfArray2 * sizeof(double)));
}
#endif
