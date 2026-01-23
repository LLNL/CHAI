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
#include "umpire/ResourceManager.hpp"
#include "umpire/TypedAllocator.hpp"
#include <cstddef>
#include <vector>

namespace chai::expt
{
  template <typename ElementType>
  class HostArrayManager {
    public:
      /*!
       * \brief Default-constructs a HostArrayManager with zero elements
       *        and a default allocator for host memory allocations.
       */
      HostArrayManager() = default;

      /*!
       * \brief Constructs a HostArrayManager with zero elements
       *        and \p allocator for host memory allocations.
       *
       * \param allocator Allocator used for host memory allocations.
       */
      explicit HostArrayManager(const umpire::Allocator& allocator)
        : m_storage{std::vector<ElementType, ::umpire::TypedAllocator<ElementType>>(::umpire::TypedAllocator<ElementType>(allocator))}
      {
      }

      /*!
       * \brief Constructs a HostArrayManager with \p size elements
       *        using the default allocator for host memory allocations.
       *
       * \param size Number of elements to allocate.
       */
      explicit HostArrayManager(std::size_t size)
        : m_storage{std::vector<ElementType, ::umpire::TypedAllocator<ElementType>>(size, ::umpire::TypedAllocator<ElementType>(::umpire::ResourceManager::getInstance().getAllocator("HOST")))}
      {
      }

      /*!
       * \brief Constructs a HostArrayManager with \p size elements
       *        using \p allocator for host memory allocations.
       *
       * \param size Number of elements to allocate.
       * \param allocator Allocator used for host memory allocations.
       */
      HostArrayManager(std::size_t size,
                       const umpire::Allocator& allocator)
        : m_storage{std::vector<ElementType, ::umpire::TypedAllocator<ElementType>>(size, ::umpire::TypedAllocator<ElementType>(allocator))}
      {
      }

      void resize(std::size_t new_size)
      {
        m_storage.resize(new_size);
      }

      std::size_t size() const
      {
        return m_storage.size();
      }

      ElementType* data()
      {
        return m_storage.empty() ? nullptr : m_storage.data();
      }

    private:
      std::vector<ElementType, ::umpire::TypedAllocator<ElementType>> m_storage{::umpire::TypedAllocator<ElementType>(::umpire::ResourceManager::getInstance().getAllocator("HOST"))};
  };  // class HostArrayManager
}  // namespace chai::expt

#endif  // CHAI_HOST_ARRAY_MANAGER_HPP
