//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ExecutionContext.hpp"
#include "chai/expt/ExectuionContextManager.hpp"

#ifndef CHAI_EXECUTION_CONTEXT_GUARD_HPP
#define CHAI_EXECUTION_CONTEXT_GUARD_HPP

namespace chai {
namespace expt {
  class ExecutionContextGuard {
    public:
      explicit ExecutionContextGuard(ExecutionContext executionContext) {
        m_execution_context_manager.setExecutionContext(executionContext);
      }

      ~ExecutionContextGuard() {
        m_execution_context_manager.setExecutionContext(m_last_execution_context);
      }

    private:
      ExecutionContextManager& m_execution_context_manager{ExecutionContextManager::getInstance()};
      ExecutionContext m_last_execution_context{m_execution_context_manager.getExecutionContext()};
  };
}  // namespace expt
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_GUARD_HPP