//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"

namespace chai::expt {
  void ContextRAJAPlugin::preCapture(const ::RAJA::util::PluginContext& p) {
    Context context = Context::NONE;

    switch (p.platform) {
      case ::RAJA::Platform::host:
        context = Context::HOST;
        break;
#if defined(CHAI_ENABLE_CUDA)
      case ::RAJA::Platform::cuda:
        context = Context::DEVICE;
        break;
#endif
#if defined(CHAI_ENABLE_HIP)
      case ::RAJA::Platform::hip:
        context = Context::DEVICE;
        break;
#endif
      default:
        context = Context::NONE;
        break;
    }

    ContextManager::getInstance().setContext(context);
  }

  void ContextRAJAPlugin::postCapture(const ::RAJA::util::PluginContext&) {
    ContextManager::getInstance().setContext(Context::NONE);
  }
}  // namespace chai::expt
