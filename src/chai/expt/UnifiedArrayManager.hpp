//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_UNIFIED_ARRAY_MANAGER_HPP
#define CHAI_UNIFIED_ARRAY_MANAGER_HPP

#include "umpire/ResourceManager.hpp"
#include "umpire/TypedAllocator.hpp"
#include <cstddef>
#include <vector>

namespace chai::expt
{
  /*!
   * \brief This class manages a host array. It is designed for use with
   *        ManagedArrayPointer.
   *
   * \tparam ElementType The type of elements contained in this array.
   *
   * \note UnifiedArrayManager performs value initialization of each array element.
   *       That is to say, numeric types will be initialized to zero and nontrivial
   *       types will be default constructed. In the future, this behavior may
   *       change to default initialization for performance reasons, such that
   *       numeric types will be left in an indeterminate state and nontrivial
   *       types will be default constructed.
   *
   * \note UnifiedArrayManager does not rely on ContextManager, so it will behave
   *       differently than other array managers. The major difference is that
   *       when used with ManagedArrayPointer, the ManagedArrayPointer does not
   *       need the update method called or to be copy constructed before it can
   *       be used on the host. Since it will not respect the current Context,
   *       be extra careful to avoid using it on the device.
   */
  template <typename ElementType>
  class UnifiedArrayManager {
    private:
      /*!
       * \brief Allocator used by the managed host storage.
       */
      using AllocatorType = ::umpire::TypedAllocator<ElementType>;

      /*!
       * \brief Underlying contiguous host storage type for managed elements.
       */
      using StorageType = std::vector<ElementType, AllocatorType>;

    public:
      /*!
       * \brief Default-constructs a UnifiedArrayManager with zero elements
       *        and a default allocator for host memory allocations.
       */
      UnifiedArrayManager() = default;

      /*!
       * \brief Constructs a UnifiedArrayManager with zero elements
       *        and \p allocator for host memory allocations.
       *
       * \param allocator Allocator used for host memory allocations.
       */
      explicit UnifiedArrayManager(const umpire::Allocator& allocator)
        : m_storage{StorageType(AllocatorType(allocator))}
      {
      }

      /*!
       * \brief Constructs a UnifiedArrayManager with \p size elements
       *        using the default allocator for host memory allocations.
       *
       * \param size Number of elements to allocate.
       */
      explicit UnifiedArrayManager(std::size_t size)
        : m_storage{StorageType(size, AllocatorType(::umpire::ResourceManager::getInstance().getAllocator("HOST")))}
      {
      }

      /*!
       * \brief Constructs a UnifiedArrayManager with \p size elements
       *        using \p allocator for host memory allocations.
       *
       * \param size Number of elements to allocate.
       * \param allocator Allocator used for host memory allocations.
       */
      UnifiedArrayManager(std::size_t size,
                       const umpire::Allocator& allocator)
        : m_storage{StorageType(size, AllocatorType(allocator))}
      {
      }

      /*!
       * \brief Resizes the managed storage to \p new_size elements.
       *
       * \param new_size New number of elements.
       */
      void resize(std::size_t new_size)
      {
        m_storage.resize(new_size);
      }

      /*!
       * \brief Returns the number of elements currently managed.
       *
       * \return Number of elements in the managed storage.
       */
      std::size_t size() const
      {
        return m_storage.size();
      }

      /*!
       * \brief Returns a pointer to the underlying contiguous storage.
       *
       * \return Pointer to the first element, or nullptr if the storage is empty.
       */
      ElementType* data(bool touch)
      {
        ContextManager& contextManager = ContextManager::getInstance();
        Context context = contextManager.getContext();

        if (context != m_modified)
        {
          contextManager.synchronize(m_modified);
        }

        if (touch)
        {
          m_modified = context;
        }

        return m_storage.empty() ? nullptr : m_storage.data();
      }

    private:
      /*!
       * \brief Underlying host storage for the managed elements.
       */
      StorageType m_storage{AllocatorType(::umpire::ResourceManager::getInstance().getAllocator("UM"))};

      /*!
       * \brief Context in which the managed storage was most recently modified.
       *
       * \note Used to determine when synchronization is required before accessing
       *       the underlying storage from the current context.
       */
      Context m_modified{Context::NONE};
  };  // class UnifiedArrayManager
}  // namespace chai::expt

#endif  // CHAI_UNIFIED_ARRAY_MANAGER_HPP
