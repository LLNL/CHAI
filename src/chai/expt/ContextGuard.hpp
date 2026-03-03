//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_GUARD_HPP
#define CHAI_CONTEXT_GUARD_HPP

#include "chai/expt/ExecutionContext.hpp"
#include "chai/expt/ContextManager.hpp"

namespace chai::expt {
  /*!
   * \brief RAII guard that temporarily sets the active execution context and
   *        restores the previously active context upon destruction.
   */
  class ContextGuard {
    public:
      /*!
       * \brief Sets the active execution context for the lifetime of this guard.
       * \param context The execution context to set as active.
       */
      explicit ContextGuard(ExecutionContext context) {
        m_context_manager.setContext(context);
      }

      /*!
       * \brief Restores the execution context that was active when this guard was created.
       */
      ~ContextGuard() {
        if (m_saved_context)
        {
          m_context_manager.setContext(*m_saved_context);
        }
        else
        {
          m_context_manager.clearContext();
        }
      }

    private:
      /*!
       * \brief Reference to the global ContextManager instance.
       */
      ContextManager& m_context_manager{ContextManager::getInstance()};

      /*!
       * Context that was active at guard construction time.
       */
      OptionalExecutionContext m_saved_context{m_context_manager.getContext()};
  };  // class ContextGuard
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_GUARD_HPP
