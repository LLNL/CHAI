//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_DUAL_MEMORY_MANAGER_HPP
#define CHAI_DUAL_MEMORY_MANAGER_HPP

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include <cstddef>

namespace chai::expt
{
  template <typename T>
  class DualMemoryManager {
    public:
      /*!
       * \brief Default-constructs a DualMemoryManager with zero size and default host/device allocators.
       */
      DualMemoryManager() = default;

      /*!
       * \brief Constructs a DualMemoryManager with zero size using the provided host and device allocators.
       *
       * \param host_allocator Allocator used for host memory allocations.
       * \param device_allocator Allocator used for device memory allocations.
       */
      DualMemoryManager(const umpire::Allocator& host_allocator,
                        const umpire::Allocator& device_allocator)
        : m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
      }

      /*!
       * \brief Constructs a DualMemoryManager with the given size using default host/device allocators.
       *
       * \param size Number of elements to allocate.
       */
      explicit DualMemoryManager(std::size_t size)
      {
        resize(size);
      }

      /*!
       * \brief Constructs a DualMemoryManager with the given size using the provided host and device allocators.
       *
       * \param size Number of elements to allocate.
       * \param host_allocator Allocator used for host memory allocations.
       * \param device_allocator Allocator used for device memory allocations.
       */
      DualMemoryManager(std::size_t size,
                        const umpire::Allocator& host_allocator,
                        const umpire::Allocator& device_allocator)
        : m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
        resize(size);
      }

      /*!
       * \brief Move-constructs a DualMemoryManager, transferring ownership of any allocated host/device buffers.
       *
       * After construction, \p other is left in a valid, empty state.
       *
       * \param other Manager to move from.
       */
      DualMemoryManager(DualMemoryManager&& other)
        : m_host_data{other.m_host_data},
          m_device_data{other.m_device_data},
          m_size{other.m_size},
          m_modified{other.m_modified},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator}
      {
        other.m_host_data = nullptr;
        other.m_device_data = nullptr;
        other.m_size = 0;
        other.m_modified = Context::NONE;
      }

      /*!
       * \brief Destructor.
       *
       * Deallocates any host/device buffers owned by this manager.
       */
      ~DualMemoryManager()
      {
        m_host_allocator.deallocate(m_host_data);
        m_device_allocator.deallocate(m_device_data);
      }

      /*!
       * \brief Move-assigns a DualMemoryManager, transferring ownership of any allocated host/device buffers.
       *
       * Any currently-owned buffers are deallocated prior to taking ownership of \p other.
       * After assignment, \p other is left in a valid, empty state.
       *
       * \param other Manager to move from.
       *
       * \return Reference to this manager.
       */
      DualMemoryManager& operator=(DualMemoryManager&& other)
      {
        if (&other != this)
        {
          m_host_allocator.deallocate(m_host_data);
          m_device_allocator.deallocate(m_device_data);

          m_host_data = other.m_host_data;
          m_device_data = other.m_device_data;
          m_size = other.m_size;
          m_modified = other.m_modified;
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;

          other.m_host_data = nullptr;
          other.m_device_data = nullptr;
          other.m_size = 0;
          other.m_modified = Context::NONE;
        }

        return *this;
      }

      /*!
       * \brief Resizes the managed allocation to \p new_size elements.
       *
       * The resize preserves the existing contents up to \c min(old_size, new_size).
       * Reallocation is performed in the "authoritative" memory space:
       * - If the host copy is authoritative (\c m_modified == Context::HOST), or only a host allocation exists,
       *   the host allocation is resized and any device allocation is discarded.
       * - Otherwise, the device allocation is resized and any host allocation is discarded.
       *
       * After resizing, \c m_modified is set to the space that was resized (HOST or DEVICE).
       *
       * \param new_size New number of elements.
       */
      void resize(std::size_t new_size)
      {
        if (new_size != m_size)
        {
          std::size_t old_size_bytes = m_size   * sizeof(T);
          std::size_t new_size_bytes = new_size * sizeof(T);

          if (m_modified == Context::HOST ||
              (m_host_data && !m_device_data))
          {
            if (m_device_data)
            {
              m_device_allocator.deallocate(m_device_data);
              m_device_data = nullptr;
            }

            T* new_host_data = nullptr;

            if (new_size > 0)
            {
              new_host_data = static_cast<T*>(m_host_allocator.allocate(new_size_bytes));
            }

            if (m_host_data)
            {
              umpire::ResourceManager::getInstance().copy(m_host_data, new_host_data, std::min(old_size_bytes, new_size_bytes));
              m_host_allocator.deallocate(m_host_data);
            }

            m_host_data = new_host_data;
            m_modified = Context::HOST;
          }
          else
          {
            if (m_host_data)
            {
              m_host_allocator.deallocate(m_host_data);
              m_host_data = nullptr;
            }

            T* new_device_data = nullptr;

            if (new_size > 0)
            {
              new_device_data = static_cast<T*>(m_device_allocator.allocate(new_size_bytes));
            }

            if (m_device_data)
            {
              umpire::ResourceManager::getInstance().copy(m_device_data, new_device_data, std::min(old_size_bytes, new_size_bytes));
              m_device_allocator.deallocate(m_device_data);
            }

            m_device_data = new_device_data;
            m_modified = Context::DEVICE;
          }

          m_size = new_size;
        }
      }

      /*!
       * \brief Returns the number of elements managed by this DualMemoryManager.
       *
       * \return Current size (in elements).
       */
      std::size_t size() const
      {
        return m_size;
      }

      /*!
       * \brief Convenience overload that returns a pointer to the managed data in the current context.
       *
       * Uses the active context provided by ContextManager::getInstance().getContext() and forwards
       * to data(Context, bool) to perform any required allocation and synchronization.
       *
       * \param touch Whether the caller intends to modify the returned data.
       *
       * \return Pointer to data in the current context, or nullptr if the current context is Context::NONE.
       */
      T* data(bool touch)
      {
        return data(ContextManager::getInstance().getContext(), touch);
      }

    private:
      /*!
       * \brief Pointer to data in the HOST context.
       */
      T* m_host_data{nullptr};

      /*!
       * \brief Pointer to data in the DEVICE context.
       */
      T* m_device_data{nullptr};

      /*!
       * \brief Number of elements currently managed by this DualMemoryManager.
       */
      std::size_t m_size{0};

      /*!
       * \brief Indicates which memory context (HOST or DEVICE) currently holds the authoritative
       *        (most recently modified) copy of the data.
       *
       * Context::NONE indicates that neither context is considered authoritative.
       */
      Context m_modified{Context::NONE};

      /*!
       * \brief Allocator used for host memory allocations.
       */
      umpire::Allocator m_host_allocator{umpire::ResourceManager::getInstance().getAllocator("HOST")};

      /*!
       * \brief Allocator used for device memory allocations.
       *
       * If CHAI is built with CUDA or HIP enabled, this defaults to the "DEVICE" allocator.
       * Otherwise, it falls back to the "HOST" allocator.
       */
      umpire::Allocator m_device_allocator{
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
        umpire::ResourceManager::getInstance().getAllocator("DEVICE")
#else
        umpire::ResourceManager::getInstance().getAllocator("HOST")
#endif
      };

      /*!
       * \brief Returns a pointer to the managed data in the requested memory context, performing
       *        allocation and synchronization as needed.
       *
       * If the requested context buffer is not yet allocated, it is allocated. If the other context
       * currently holds the authoritative (most recently modified) copy, the data is copied to the
       * requested context before returning.
       *
       * The \p touch parameter controls whether the returned pointer is considered to be modified:
       * - If \p touch is true, the requested context becomes authoritative (\c m_modified is set to
       *   Context::HOST or Context::DEVICE).
       * - If \p touch is false, no context is considered authoritative (\c m_modified is set to
       *   Context::NONE).
       *
       * \param context The desired context (HOST or DEVICE).
       * \param touch Whether the caller intends to modify the returned data.
       *
       * \return Pointer to data in the requested context, or nullptr if \p context is Context::NONE.
       */
      T* data(Context context, bool touch = true)
      {
        if (context == Context::DEVICE)
        {
          if (m_device_data == nullptr)
          {
            m_device_data = static_cast<T*>(m_device_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::HOST)
          {
            umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
          }

          if (touch)
          {
            m_modified = Context::DEVICE;
          }
          else
          {
            m_modified = Context::NONE;
          }

          return m_device_data;
        }
        else if (context == Context::HOST)
        {
          if (m_host_data == nullptr)
          {
            m_host_data = static_cast<T*>(m_host_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::DEVICE)
          {
            umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
          }

          if (touch)
          {
            m_modified = Context::HOST;
          }
          else
          {
            m_modified = Context::NONE;
          }

          return m_host_data;
        }
        else
        {
          return nullptr;
        }
      }
  };  // class DualMemoryManager
}  // namespace chai::expt

#endif  // CHAI_DUAL_MEMORY_MANAGER_HPP