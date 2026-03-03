//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_MANAGER_HPP
#define CHAI_CONTEXT_MANAGER_HPP

#include "chai/config.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "camp/helpers.hpp"

#include <cstdint>
#include <type_traits>
#include <unordered_map>

namespace chai::expt {
  /*!
   * \brief Singleton class for managing the current context
   *        and context synchronization across the application.
   */
  class ContextManager
  {
    public:
      /*!
       * \brief Get the singleton instance.
       */
      static ContextManager& getInstance()
      {
        static ContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Disable copy construction.
       *
       * ContextManager is a singleton and must not be copied.
       */
      ContextManager(const ContextManager&) = delete;

      /*!
       * \brief Disable copy assignment.
       *
       * ContextManager is a singleton and must not be assigned.
       */
      ContextManager& operator=(const ContextManager&) = delete;

      /*!
       * \brief Get the current execution context for the calling thread.
       */
      OptionalExecutionContext getContext() const
      {
        return s_context;
      }

      /*!
       * \brief Query whether a context is currently set for this thread.
       */
      bool hasContext() const
      {
        return s_context.has_value();
      }

      /*!
       * \brief Clear the current context for the calling thread.
       */
      void clearContext()
      {
        s_context.reset();
      }

      /*!
       * \brief Set the current context for the calling thread.
       *
       * Setting a device context marks that stream as not synchronized.
       */
      void setContext(ExecutionContext context)
      {
        s_context = std::move(context);

        std::visit(
          [this](const auto& ctx) {
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HostContext>)
            {
              (void)ctx;
            }

#if defined(CHAI_ENABLE_CUDA)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, CudaContext>)
            {
              s_stream_synchronized[StreamKey::cuda(ctx.stream)] = false;
            }
#endif

#if defined(CHAI_ENABLE_HIP)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HipContext>)
            {
              s_stream_synchronized[StreamKey::hip(ctx.stream)] = false;
            }
#endif
          },
          *s_context);
      }

      /*!
       * \brief Synchronize the current context (no-op if already synchronized).
       *
       * For device contexts, synchronization is stream-aware (stream synchronize).
       */
      void synchronize()
      {
        if (!s_context)
        {
          return;
        }

        std::visit(
          [this](const auto& ctx) {
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HostContext>)
            {
              return;
            }

#if defined(CHAI_ENABLE_CUDA)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, CudaContext>)
            {
              const StreamKey key = StreamKey::cuda(ctx.stream);
              if (s_stream_synchronized[key])
              {
                return;
              }

              CAMP_CUDA_API_INVOKE_AND_CHECK(cudaStreamSynchronize, ctx.stream);

              s_stream_synchronized[key] = true;
            }
#endif

#if defined(CHAI_ENABLE_HIP)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HipContext>)
            {
              const StreamKey key = StreamKey::hip(ctx.stream);
              if (s_stream_synchronized[key])
              {
                return;
              }

              CAMP_HIP_API_INVOKE_AND_CHECK(hipStreamSynchronize, ctx.stream);

              s_stream_synchronized[key] = true;
            }
#endif
          },
          *s_context);
      }

      /*!
       * \brief Query whether the current context is synchronized.
       */
      bool isSynchronized() const
      {
        if (!s_context)
        {
          return true;
        }

        return std::visit(
          [this](const auto& ctx) -> bool {
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HostContext>)
            {
              return true;
            }

#if defined(CHAI_ENABLE_CUDA)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, CudaContext>)
            {
              return s_stream_synchronized[StreamKey::cuda(ctx.stream)];
            }
#endif

#if defined(CHAI_ENABLE_HIP)
            if constexpr (std::is_same_v<std::decay_t<decltype(ctx)>, HipContext>)
            {
              return s_stream_synchronized[StreamKey::hip(ctx.stream)];
            }
#endif

            return true;
          },
          *s_context);
      }

      /*!
       * \brief Reset manager state to defaults.
       */
      void reset()
      {
        clearContext();
        s_stream_synchronized.clear();
      }

    private:
      enum class Backend
      {
        CUDA,
        HIP
      };

      struct StreamKey
      {
        Backend backend{};
        std::uintptr_t handle{};

#if defined(CHAI_ENABLE_CUDA)
        static StreamKey cuda(cudaStream_t stream)
        {
          return StreamKey{Backend::CUDA, reinterpret_cast<std::uintptr_t>(stream)};
        }
#endif

#if defined(CHAI_ENABLE_HIP)
        static StreamKey hip(hipStream_t stream)
        {
          return StreamKey{Backend::HIP, reinterpret_cast<std::uintptr_t>(stream)};
        }
#endif
      };

      struct StreamKeyHash
      {
        std::size_t operator()(const StreamKey& key) const noexcept
        {
          const std::size_t backend_hash = std::hash<int>{}(static_cast<int>(key.backend));
          const std::size_t handle_hash = std::hash<std::uintptr_t>{}(key.handle);
          return backend_hash ^ (handle_hash + 0x9e3779b97f4a7c15ULL + (backend_hash << 6U) + (backend_hash >> 2U));
        }
      };

      struct StreamKeyEq
      {
        bool operator()(const StreamKey& a, const StreamKey& b) const noexcept
        {
          return a.backend == b.backend && a.handle == b.handle;
        }
      };

      /*!
       * \brief Default constructor.
       *
       * Private to enforce singleton access via getInstance().
       */
      ContextManager() = default;

      /*!
       * \brief Current context for the calling thread.
       */
      inline static thread_local OptionalExecutionContext s_context{};

      /*!
       * \brief Synchronization state per device stream.
       */
      inline static thread_local std::unordered_map<StreamKey, bool, StreamKeyHash, StreamKeyEq> s_stream_synchronized{};
  };  // class ContextManager
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_MANAGER_HPP
