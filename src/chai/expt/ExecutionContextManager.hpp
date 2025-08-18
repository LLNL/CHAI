//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_MANAGER_HPP
#define CHAI_EXECUTION_CONTEXT_MANAGER_HPP

#include "chai/expt/ExecutionContext.hpp"

namespace chai {
namespace expt {
  /*!
   * \class ExecutionContextManager
   *
   * \brief Singleton class for managing the current execution context.
   *
   * This class provides a centralized way to get and set the current execution
   * context across the application.
   */
  class ExecutionContextManager {
    public:
      /*!
       * \brief Get the singleton instance of ExecutionContextManager.
       *
       * \return The singleton instance.
       */
      static ExecutionContextManager& getInstance() {
        static inline ExecutionContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Private copy constructor to prevent copying.
       */
      ExecutionContextManager(const ExecutionContextManager&) = delete;

      /*!
       * \brief Private assignment operator to prevent assignment.
       */
      ExecutionContextManager& operator=(const ExecutionContextManager&) = delete;

      /*!
       * \brief Get the current execution context.
       *
       * \return The current context.
       */
      ExecutionContext getContext() const {
        return m_current_context;
      }

      /*!
       * \brief Set the current execution context.
       *
       * \param context The new context to set.
       */
      void setContext(ExecutionContext context) {
        m_current_context = context;
      }

    private:
      /*!
       * \brief Private constructor for singleton pattern.
       */
      constexpr ExecutionContextManager() noexcept = default;

      /*!
       * \brief The current execution context.
       */
      ExecutionContext m_current_context = NONE;
  };  // class ExecutionContextManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_MANAGER_HPP
