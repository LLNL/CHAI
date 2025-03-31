//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "chai/config.hpp"
#include "chai/expt/CopyHidingManager.hpp"
#include "umpire/ResourceManager.hpp"

class CopyHidingManagerTest : public testing::Test {
  protected:
    int m_host_allocator_id{umpire::ResourceManager::getInstance().getAllocator("HOST").getId()};
    int m_device_allocator_id{umpire::ResourceManager::getInstance().getAllocator("DEVICE").getId()};
    std::size_t m_size{100};
    chai::expt::CopyHidingManager m_manager{m_host_allocator_id,
                                            m_device_allocator_id,
                                            m_size};
};


TEST_F(CopyHidingManagerTest, HostAllocatorID)
{
  EXPECT_EQ(m_host_allocator_id, m_manager.getHostAllocatorID());
}

TEST_F(CopyHidingManagerTest, DeviceAllocatorID)
{
  EXPECT_EQ(m_device_allocator_id, m_manager.getDeviceAllocatorID());
}

TEST_F(CopyHidingManagerTest, Size)
{
  EXPECT_EQ(m_size, m_manager.size());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextNone)
{
  EXPECT_EQ(nullptr, m_manager.data(chai::expt::ExecutionContext::NONE, false));
  EXPECT_EQ(chai::expt::ExecutionContext::NONE, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextHost)
{
  EXPECT_NE(nullptr, m_manager.data(chai::expt::ExecutionContext::HOST, false));
  EXPECT_EQ(chai::expt::ExecutionContext::NONE, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextDevice)
{
  EXPECT_NE(nullptr, m_manager.data(chai::expt::ExecutionContext::DEVICE, false));
  EXPECT_EQ(chai::expt::ExecutionContext::NONE, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, Touch)
{
  EXPECT_EQ(chai::expt::ExecutionContext::NONE, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextNoneTouch)
{
  EXPECT_EQ(nullptr, m_manager.data(chai::expt::ExecutionContext::NONE, true));
  EXPECT_EQ(chai::expt::ExecutionContext::NONE, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextHostTouch)
{
  EXPECT_NE(nullptr, m_manager.data(chai::expt::ExecutionContext::HOST, true));
  EXPECT_EQ(chai::expt::ExecutionContext::HOST, m_manager.getTouch());
}

TEST_F(CopyHidingManagerTest, DataExecutionContextDeviceTouch)
{
  EXPECT_NE(nullptr, m_manager.data(chai::expt::ExecutionContext::DEVICE, true));
  EXPECT_EQ(chai::expt::ExecutionContext::DEVICE, m_manager.getTouch());
}
