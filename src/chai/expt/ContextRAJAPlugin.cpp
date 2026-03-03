//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"

namespace chai::expt {
  void ContextRAJAPlugin::preCapture(const ::RAJA::util::PluginContext& p) {
    switch (p.platform) {
      case ::RAJA::Platform::host:
        ContextManager::getInstance().setContext(HostContext{});
        break;
#if defined(CHAI_ENABLE_CUDA)
      case ::RAJA::Platform::cuda:
        ContextManager::getInstance().setContext(CudaContext{});
        break;
#endif
#if defined(CHAI_ENABLE_HIP)
      case ::RAJA::Platform::hip:
        ContextManager::getInstance().setContext(HipContext{});
        break;
#endif
      default:
        ContextManager::getInstance().clearContext();
        break;
    }
  }

  void ContextRAJAPlugin::postCapture(const ::RAJA::util::PluginContext&) {
    ContextManager::getInstance().clearContext();
  }
}  // namespace chai::expt
