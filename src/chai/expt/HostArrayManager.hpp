//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_HOST_ARRAY_MANAGER_HPP
#define CHAI_HOST_ARRAY_MANAGER_HPP

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include <cstddef>

namespace chai::expt
{
  class HostArrayManager {
    public:
      /*!
       * \brief Default-constructs a HostArrayManager with zero size and default host allocator.
       */
      HostArrayManager() = default;

      /*!
       * \brief Constructs a HostArrayManager with zero size and the provided host allocator.
       *
       * \param allocator Allocator used for host memory allocations.
       */
      HostArrayManager(const umpire::Allocator& allocator)
        : m_allocator{host_allocator}
      {
      }

      /*!
       * \brief Constructs a HostArrayManager with the given size using the default host allocator.
       *
       * \param bytes Number of bytes to allocate.
       */
      explicit HostArrayManager(std::size_t bytes)
      {
        resize_bytes(bytes);
      }

      /*!
       * \brief Constructs a HostArrayManager with the given size using the provided host allocator.
       *
       * \param bytes Number of bytes to allocate.
       * \param allocator Allocator used for host memory allocations.
       */
      HostArrayManager(std::size_t bytes,
                       const umpire::Allocator& allocator)
        : m_allocator{allocator}
      {
        resize_bytes(bytes);
      }

      /*!
       * \brief Move-constructs a HostArrayManager, transferring ownership of host buffer (if allocated).
       *
       * After construction, \p other is left in a valid, empty state.
       *
       * \param other Manager to move from.
       */
      HostArrayManager(HostArrayManager&& other)
        : m_data{other.m_data},
          m_size_bytes{other.m_size_bytes},
          m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size_bytes = 0;
      }

      /*!
       * \brief Destructor.
       *
       * Deallocates host buffer owned by this manager.
       */
      ~HostArrayManager()
      {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Move-assigns a HostArrayManager, transferring ownership of any allocated host/device buffers.
       *
       * Any currently-owned buffers are deallocated prior to taking ownership of \p other.
       * After assignment, \p other is left in a valid, empty state.
       *
       * \param other Manager to move from.
       *
       * \return Reference to this manager.
       */
      HostArrayManager& operator=(HostArrayManager&& other)
      {
        if (&other != this)
        {
          m_allocator.deallocate(m_data);

          m_data = other.m_data;
          m_size_bytes = other.m_size_bytes;
          m_allocator = other.m_allocator;

          other.m_data = nullptr;
          other.m_size_bytes = 0;
        }

        return *this;
      }

      /*!
       * \brief Resizes the managed allocation to \p new_bytes bytes.
       *
       * The resize preserves the existing contents up to \c min(old_size, new_size).
       * Reallocation is performed in the "authoritative" memory space:
       * - If the host copy is authoritative (\c m_modified == Context::HOST), or only a host allocation exists,
       *   the host allocation is resized and any device allocation is discarded.
       * - Otherwise, the device allocation is resized and any host allocation is discarded.
       *
       * After resizing, \c m_modified is set to the space that was resized (HOST or DEVICE).
       *
       * \param new_size_bytes New number of bytes.
       */
      void resize_bytes(std::size_t new_size_bytes)
      {
        if (new_size != m_size_bytes)
        {
          void* new_data = m_allocator.allocate(new_size_bytes);
          ::umpire::ResourceManager::getInstance().copy(m_data, new_data, std::min(m_size_bytes, new_size_bytes));
          m_size_bytes = new_size_bytes;
          m_allocator.deallocate(m_data);
          m_data = new_data;
        }
      }

      /*!
       * \brief Returns the number of bytes managed by this HostArrayManager.
       *
       * \return Current size (in bytes).
       */
      std::size_t size_bytes() const
      {
        return m_size_bytes;
      }

      /*!
       * \brief Returns a pointer to the managed host data if the current context is HOST.
       *
       * If the current context is not HOST, returns \c nullptr.
       *
       * \return Pointer to host data when in HOST context; otherwise \c nullptr.
       */
      T* data()
      {
        Context context = ContextManager::getInstance().getContext();

        if (context == Context::HOST)
        {
          return m_data;
        }
        else
        {
          return nullptr;
        }
      }

    private:
      /*!
       * \brief Pointer to data in the HOST context.
       */
      T* m_data{nullptr};

      /*!
       * \brief Number of bytes currently managed by this HostArrayManager.
       */
      std::size_t m_size_bytes{0};

      /*!
       * \brief Allocator used for host memory allocations.
       */
      umpire::Allocator m_allocator{umpire::ResourceManager::getInstance().getAllocator("HOST")};
  };  // class HostArrayManager
}  // namespace chai::expt

#endif  // CHAI_HOST_ARRAY_MANAGER_HPP