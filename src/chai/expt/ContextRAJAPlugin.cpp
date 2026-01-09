//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"

namespace chai::expt {
  void ContextRAJAPlugin::preCapture(const ::RAJA::util::PluginContext& p)
  {
    switch (p.platform) {
      case ::RAJA::Platform::host:
        m_context_manager.setContext(Context::HOST);
        break;
#if defined(CHAI_ENABLE_CUDA)
      case ::RAJA::Platform::cuda:
        m_context_manager.setContext(Context::DEVICE);
        break;
#endif
#if defined(CHAI_ENABLE_HIP)
      case ::RAJA::Platform::hip:
        m_context_manager.setContext(Context::DEVICE);
        break;
#endif
      default:
        m_context_manager.setContext(Context::NONE);
        break;
    }
  }

  void ContextRAJAPlugin::postCapture(const ::RAJA::util::PluginContext&)
  {
    m_context_manager.setContext(Context::NONE);
  }
}

// Pre-main registration of plugin with RAJA
static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
  "CHAIContextPlugin",
  "Plugin that integrates CHAI context management with RAJA.");
