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

TEST(HostManager, Constructor)
{
  size_t size = 100;
  int allocatorID = umpire::ResourceManager::getInstance().getAllocator("HOST").getId();

  chai::expt::HostManager manager(allocatorID, size);

  ASSERT_EQ(allocatorID, manager.getAllocatorID());
  ASSERT_EQ(size, manager.size());

  void* data;

  /*
   * Test chai::expt::ExecutionContext::NONE
   */
  manager.execution_context() = chai::expt::ExecutionContext::NONE;
  manager.update(data, false);
  ASSERT_EQ(nullptr, data);

  /*
   * Test chai::expt::ExecutionContext::HOST
   */
  manager.execution_context() = chai::expt::ExecutionContext::HOST;
  manager.update(data, false);
  ASSERT_NE(nullptr, data);

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*
   * Test chai::expt::ExecutionContext::DEVICE
   */
  manager.execution_context() = chai::expt::ExecutionContext::DEVICE;
  manager.update(data, false);
  ASSERT_EQ(nullptr, data);
#endif

  /*
   * Reset chai::expt::ExecutionContext
   */
  manager.execution_context() = chai::expt::ExecutionContext::NONE;
}
