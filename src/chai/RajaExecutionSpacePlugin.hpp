//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_RajaExecutionSpacePlugin_HPP
#define CHAI_RajaExecutionSpacePlugin_HPP

#include "RAJA/util/PluginStrategy.hpp"

namespace chai {

class ArrayManager;

class RajaExecutionSpacePlugin :
  public RAJA::util::PluginStrategy
{
  public:
    RajaExecutionSpacePlugin();

    void preCapture(const RAJA::util::PluginContext& p) override;

    void postCapture(const RAJA::util::PluginContext& p) override;

  private:
    chai::ArrayManager* m_arraymanager{nullptr};
};

void linkRajaPlugin();

}

#endif // CHAI_RajaExecutionSpacePlugin_HPP
