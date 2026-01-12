//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_GUARD_HPP
#define CHAI_CONTEXT_GUARD_HPP

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"

namespace chai::expt {
  /*!
   * \brief RAII guard that temporarily sets the active Context and restores the
   *        previously active Context upon destruction.
   */
  class ContextGuard {
    public:
      /*!
       * \brief Sets the active Context for the lifetime of this guard.
       * \param context The Context to set as active.
       */
      explicit ContextGuard(Context context) {
        m_context_manager.setContext(context);
      }

      /*!
       * \brief Restores the Context that was active when this guard was created.
       */
      ~ContextGuard() {
        m_context_manager.setContext(m_saved_context);
      }

    private:
      /*!
       * \brief Reference to the global ContextManager instance.
       */
      ContextManager& m_context_manager{ContextManager::getInstance()};

      /*!
       * Context that was active at guard construction time.
       */
      Context m_saved_context{m_context_manager.getContext()};
  };  // class ContextGuard
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_GUARD_HPP
