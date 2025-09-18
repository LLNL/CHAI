//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_GUARD_HPP
#define CHAI_CONTEXT_GUARD_HPP

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"

namespace chai {
namespace expt {
  class ContextGuard {
    public:
      explicit ContextGuard(Context context) {
        m_context_manager.setContext(context);
      }

      ~ContextGuard() {
        m_context_manager.setContext(m_saved_context);
      }

    private:
      ContextManager& m_context_manager{ContextManager::getInstance()};
      Context m_saved_context{m_context_manager.getContext()};
  };
}  // namespace expt
}  // namespace chai

#endif  // CHAI_CONTEXT_GUARD_HPP