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
   * \brief Singleton class for managing the current context.
   *
   * This class provides a centralized way to get and set the current
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
        static ContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Deleted copy constructor to prevent copying.
       */
      ContextManager(const ContextManager&) = delete;

      /*!
       * \brief Deleted assignment operator to prevent assignment.
       */
      ContextManager& operator=(const ContextManager&) = delete;

      /*!
       * \brief Get the current context.
       *
       * \return The current context.
       */
      Context getContext() const {
        return m_context;
      }

      /*!
       * \brief Set the current context.
       *
       * \param context The new context to set.
       */
      void setContext(Context context) {
        m_context = context;
        m_synchronized[context] = false;
      }

      /*!
       * \brief Synchronize the given  context.
       *
       * \param context The context that needs synchronization.
       */
      void synchronize(Context context) {
        auto it = m_synchronized.find(context);

        if (it != m_synchronized.end()) {
          #if defined(CHAI_ENABLE_DEVICE)
          if (context == Context::DEVICE) {
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
          if (context == Context::DEVICE) {
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
       * \brief Check if a specific context needs synchronization.
       *
       * \param context The  context to check.
       * \return True if the context needs synchronization, false otherwise.
       */
      bool isSynchronized(Context context) const {
        auto it = m_synchronized.find(context);

        if (it == m_synchronized.end()) {
          return true;
        }
        else {
          return it->second;
        }
      }

      /*!
       * \brief Mark the given context as synchronized.
       *
       * This should only be called after synchronization has been performed.
       *
       * \param context The context to clear the synchronization flag for.
       */
      void markSynchronized(Context context) {
        m_synchronized[context] = true;
      }

    private:
      /*!
       * \brief Private constructor for singleton pattern.
       */
      constexpr ContextManager() noexcept = default;

      /*!
       * \brief The current context.
       */
      Context m_context = Context::NONE;

      /*!
       * \brief Map for tracking which contexts are synchronized.
       */
      std::unordered_map<Context, bool> m_synchronized;
  };  // class ContextManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_CONTEXT_MANAGER_HPP
