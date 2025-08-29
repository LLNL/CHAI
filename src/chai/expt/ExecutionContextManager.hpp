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
        static ExecutionContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Deleted copy constructor to prevent copying.
       */
      ExecutionContextManager(const ExecutionContextManager&) = delete;

      /*!
       * \brief Deleted assignment operator to prevent assignment.
       */
      ExecutionContextManager& operator=(const ExecutionContextManager&) = delete;

      /*!
       * \brief Get the current execution context.
       *
       * \return The current context.
       */
      ExecutionContext getExecutionContext() const {
        return m_execution_context;
      }

      /*!
       * \brief Set the current execution context.
       *
       * \param context The new context to set.
       */
      void setExecutionContext(ExecutionContext context) {
        m_execution_context = context;
        m_synchronized[context] = false;
      }

      /*!
       * \brief Synchronize the given execution context.
       *
       * \param context The execution context that needs synchronization.
       */
      void synchronize(ExecutionContext context) {
        auto it = m_synchronized.find(context);

        if (it != m_synchronized.end()) {
          #if defined(CHAI_ENABLE_DEVICE)
          if (context == ExecutionContext::DEVICE) {
#if defined(CHAI_ENABLE_CUDA)
            cudaDeviceSynchronize();
#elif defined(CHAI_ENABLE_HIP)
            hipDeviceSynchronize();
#endif
          }
        }
        bool& unsynchronized = m_unsynchronized[context];

        if (unsynchronized) {
#if defined(CHAI_ENABLE_DEVICE)
          if (context == ExecutionContext::DEVICE) {
#if defined(CHAI_ENABLE_CUDA)
            cudaDeviceSynchronize();
#elif defined(CHAI_ENABLE_HIP)
            hipDeviceSynchronize();
#endif
          }

          unsynchronized = false;
        }
      }

      /*!
       * \brief Check if a specific execution context needs synchronization.
       *
       * \param context The execution context to check.
       * \return True if the context needs synchronization, false otherwise.
       */
      bool isSynchronized(ExecutionContext context) const {
        auto it = m_synchronized.find(context);

        if (it == m_synchronized.end()) {
          return true;
        }
        else {
          return it->second;
        }
      }

      /*!
       * \brief Mark the given execution context as synchronized.
       *
       * This should only be called after synchronization has been performed.
       *
       * \param context The execution context to clear the synchronization flag for.
       */
      void markSynchronized(ExecutionContext context) {
        m_synchronized[context] = true;
      }

    private:
      /*!
       * \brief Private constructor for singleton pattern.
       */
      constexpr ExecutionContextManager() noexcept = default;

      /*!
       * \brief The current execution context.
       */
      ExecutionContext m_execution_context = ExecutionContext::NONE;

      /*!
       * \brief Map for tracking which execution contexts are synchronized.
       */
      std::unordered_map<ExecutionContext, bool> m_synchronized;
  };  // class ExecutionContextManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_MANAGER_HPP
