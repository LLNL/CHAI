//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CUDA_UNIFIED_ARRAY_MANAGER_HPP
#define CHAI_CUDA_UNIFIED_ARRAY_MANAGER_HPP

#include "chai/config.hpp"
#include "chai/expt/ContextManager.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(CHAI_ENABLE_CUDA)
#include "camp/helpers.hpp"
#include <cuda_runtime.h>
#endif

namespace chai::expt
{
  namespace detail
  {
    template <typename...>
    inline constexpr bool always_false_v = false;
  }

#if defined(CHAI_ENABLE_CUDA)
  /*!
   * \brief Unified-memory array manager for CUDA with stream-aware synchronization.
   *
   * \details This manager allocates unified memory (cudaMallocManaged) and tracks
   * the last writer (host or a CUDA stream). If the data was last modified on one
   * CUDA stream, that stream is synchronized before allowing use on a different
   * stream or on the host.
   *
   * Additional performance hints can be provided via Hints (prefetching, preferred
   * location, etc.). These are opportunistic and may be ignored depending on the
   * CUDA runtime and platform.
   */
  template <typename ElementType>
  class CudaUnifiedArrayManager
  {
    public:
      enum class PreferredLocation
      {
        None,
        Host,
        Device
      };

      struct Hints
      {
        cudaStream_t allocation_stream{0};
        bool prefetch_on_resize{false};
        bool prefetch_on_use{false};

        PreferredLocation preferred_location{PreferredLocation::None};
        int preferred_device{-1};

        bool advise_read_mostly{false};
        bool advise_accessed_by_host{false};
        bool advise_accessed_by_device{false};
      };

    private:
      template <typename T>
      struct ManagedAllocator
      {
        using value_type = T;

        ManagedAllocator() = default;

        template <typename U>
        ManagedAllocator(const ManagedAllocator<U>&) noexcept
        {
        }

        T* allocate(std::size_t n)
        {
          if (n == 0)
          {
            return nullptr;
          }

          void* ptr = nullptr;
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, &ptr, n * sizeof(T), cudaMemAttachGlobal);
          return static_cast<T*>(ptr);
        }

        void deallocate(T* ptr, std::size_t) noexcept
        {
          if (ptr)
          {
            CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
          }
        }

        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;
      };

      template <typename A, typename B>
      friend struct ManagedAllocator;

      enum class LastWriter
      {
        None,
        Host,
        Stream
      };

    public:
      CudaUnifiedArrayManager() = default;

      explicit CudaUnifiedArrayManager(std::size_t size)
      {
        resize(size);
      }

      CudaUnifiedArrayManager(std::size_t size, Hints hints)
        : m_hints{std::move(hints)}
      {
        resize(size);
      }

      void setHints(Hints hints)
      {
        m_hints = std::move(hints);
        applyHints();
      }

      const Hints& getHints() const
      {
        return m_hints;
      }

      void resize(std::size_t new_size)
      {
        synchronizeForHostMutation();

        m_storage.resize(new_size);

        applyHints();

        // Resizing value-initializes new elements (host-side writes).
        m_last_writer = LastWriter::Host;
        m_last_write_stream = 0;
      }

      std::size_t size() const
      {
        return m_storage.size();
      }

      ElementType* data()
      {
        prepareForUseInCurrentContext();
        return m_storage.empty() ? nullptr : m_storage.data();
      }

      void prefetchToDevice(int device, cudaStream_t stream)
      {
        if (m_storage.empty())
        {
          return;
        }

        const int target_device = device < 0 ? currentDevice() : device;
        CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemPrefetchAsync,
                                       m_storage.data(),
                                       m_storage.size() * sizeof(ElementType),
                                       target_device,
                                       stream);
      }

      void prefetchToHost(cudaStream_t stream)
      {
        if (m_storage.empty())
        {
          return;
        }

        CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemPrefetchAsync,
                                       m_storage.data(),
                                       m_storage.size() * sizeof(ElementType),
                                       cudaCpuDeviceId,
                                       stream);
      }

      void setPreferredLocationHost()
      {
        m_hints.preferred_location = PreferredLocation::Host;
        applyHints();
      }

      void setPreferredLocationDevice(int device = -1)
      {
        m_hints.preferred_location = PreferredLocation::Device;
        m_hints.preferred_device = device;
        applyHints();
      }

    private:
      static int currentDevice()
      {
        int device = 0;
        CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetDevice, &device);
        return device;
      }

      static cudaStream_t currentCudaStream(const OptionalExecutionContext& ctx)
      {
        if (!ctx)
        {
          return 0;
        }

        if (const auto* cuda_ctx = std::get_if<CudaContext>(&*ctx))
        {
          return cuda_ctx->stream;
        }

        return 0;
      }

      static bool isCudaContext(const OptionalExecutionContext& ctx)
      {
        return ctx && std::holds_alternative<CudaContext>(*ctx);
      }

      void synchronizeForHostMutation()
      {
        if (m_last_writer == LastWriter::Stream)
        {
          ContextManager::getInstance().synchronizeStream(m_last_write_stream);
          m_last_writer = LastWriter::Host;
          m_last_write_stream = 0;
        }
      }

      void prepareForUseInCurrentContext()
      {
        ContextManager& context_manager = ContextManager::getInstance();
        const OptionalExecutionContext current = context_manager.getContext();

        const bool in_cuda_context = isCudaContext(current);
        const cudaStream_t current_stream = in_cuda_context ? currentCudaStream(current) : 0;

        if (m_last_writer == LastWriter::Stream)
        {
          if (!in_cuda_context || current_stream != m_last_write_stream)
          {
            context_manager.synchronizeStream(m_last_write_stream);
          }
        }

        if (in_cuda_context && m_hints.prefetch_on_use)
        {
          prefetchToDevice(m_hints.preferred_device, current_stream);
        }
        else if (!in_cuda_context && m_hints.prefetch_on_use)
        {
          prefetchToHost(m_hints.allocation_stream);
        }

        // Conservatively assume the caller may modify in the current context.
        if (in_cuda_context)
        {
          m_last_writer = LastWriter::Stream;
          m_last_write_stream = current_stream;
          context_manager.markStreamUnsynchronized(current_stream);
        }
        else
        {
          m_last_writer = LastWriter::Host;
          m_last_write_stream = 0;
        }
      }

      void applyHints()
      {
        if (m_storage.empty())
        {
          return;
        }

        void* ptr = static_cast<void*>(m_storage.data());
        const std::size_t bytes = m_storage.size() * sizeof(ElementType);

        if (m_hints.advise_read_mostly)
        {
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, bytes, cudaMemAdviseSetReadMostly, currentDevice());
        }

        if (m_hints.advise_accessed_by_host)
        {
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
        }

        if (m_hints.advise_accessed_by_device)
        {
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, bytes, cudaMemAdviseSetAccessedBy, currentDevice());
        }

        if (m_hints.preferred_location == PreferredLocation::Host)
        {
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        }
        else if (m_hints.preferred_location == PreferredLocation::Device)
        {
          const int device = m_hints.preferred_device < 0 ? currentDevice() : m_hints.preferred_device;
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMemAdvise, ptr, bytes, cudaMemAdviseSetPreferredLocation, device);
        }

        if (m_hints.prefetch_on_resize)
        {
          if (m_hints.preferred_location == PreferredLocation::Host)
          {
            prefetchToHost(m_hints.allocation_stream);
          }
          else if (m_hints.preferred_location == PreferredLocation::Device)
          {
            prefetchToDevice(m_hints.preferred_device, m_hints.allocation_stream);
          }
        }
      }

      std::vector<ElementType, ManagedAllocator<ElementType>> m_storage{};
      Hints m_hints{};

      LastWriter m_last_writer{LastWriter::None};
      cudaStream_t m_last_write_stream{0};
  };
#else
  template <typename ElementType>
  class CudaUnifiedArrayManager
  {
    static_assert(detail::always_false_v<ElementType>, "CudaUnifiedArrayManager requires CHAI_ENABLE_CUDA");
  };
#endif
}  // namespace chai::expt

#endif  // CHAI_CUDA_UNIFIED_ARRAY_MANAGER_HPP
