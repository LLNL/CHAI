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
TEST_F(HostManagerTest, DataExecutionContextNone)
{
  EXPECT_EQ(nullptr, m_manager.data(chai::expt::ExecutionContext::DEVICE, false));
}
#endif
