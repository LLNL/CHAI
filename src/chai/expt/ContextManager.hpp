//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_MANAGER_HPP
#define CHAI_CONTEXT_MANAGER_HPP

#include "chai/expt/Context.hpp"

namespace chai {
namespace expt {
  /*!
   * \class ContextManager
   *
   * \brief Singleton class for managing the current execution context.
   *
   * This class provides a centralized way to get and set the current execution
   * context across the application.
   */
  class ContextManager {
    public:
      /*!
       * \brief Get the singleton instance of ContextManager.
       *
       * \return The singleton instance.
       */
      static ContextManager& getInstance() {
        static inline ContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Private copy constructor to prevent copying.
       */
      ContextManager(const ContextManager&) = delete;

      /*!
       * \brief Private assignment operator to prevent assignment.
       */
      ContextManager& operator=(const ContextManager&) = delete;

      /*!
       * \brief Get the current execution context.
       *
       * \return The current context.
       */
      Context getContext() const {
        return m_current_context;
      }

      /*!
       * \brief Set the current execution context.
       *
       * \param context The new context to set.
       */
      void setContext(Context context) {
        m_current_context = context;
      }

    private:
      /*!
       * \brief Private constructor for singleton pattern.
       */
      constexpr ContextManager() noexcept = default;

      /*!
       * \brief The current execution context.
       */
      Context m_current_context = NONE;
  };
}  // namespace expt
}  // namespace chai

#endif  // CHAI_CONTEXT_HPP
  
}  // namespace expt
}  // namespace chai

#endif  // CHAI_CONTEXT_MANAGER_HPP
