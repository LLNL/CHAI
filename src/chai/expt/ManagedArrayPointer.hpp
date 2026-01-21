//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_MANAGED_ARRAY_POINTER_HPP
#define CHAI_MANAGED_ARRAY_POINTER_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include <cstddef>
#include <type_traits>

namespace chai::expt
{
  /*!
   * \brief This class provides a uniform interface for working with different types of memory
   *        across multiple backends. It has pointer semantics, meaning that copies are shallow,
   *        which allows this object to be passed by value to a CUDA or HIP kernel. When copy
   *        constructed, it queries the array manager to update the cached size and pointer.
   *        from the array manager. The array must be explicitly freed from only one of the
   *        shallow copies.
   *
   * \tparam ElementType The type of elements contained in this array.
   * \tparam ManagerType Manages the underlying memory.
   */
  template <typename ElementType, typename ManagerType>
  class ManagedArrayPointer
  {
    public:
      /*!
       * @brief Constructs a default ManagedArrayPointer.
       *
       * @details Creates a null pointer with size 0 and no associated manager.
       */
      ManagedArrayPointer() = default;

      /*!
       * @brief Constructs an ManagedArrayPointer from an existing manager.
       *
       * @param manager Pointer to a manager that owns/manages the underlying array.
       *
       * @details The ManagedArrayPointer assumes pointer ownership semantics, meaning
       * this ManagedArrayPointer or any copy of this ManagedArrayPointer can delete the manager.
       */
      explicit ManagedArrayPointer(ManagerType* manager)
        : m_manager{manager}
      {
      }

      /*!
       * @brief Copy-constructs an ManagedArrayPointer from another ManagedArrayPointer.
       *
       * @param other The ManagedArrayPointer to copy.
       *
       * @details Copies the cached pointer, size, and manager pointer. Ownership
       * semantics are preserved (the manager pointer is shared). The internal
       * cached pointer/size are synchronized by calling update().
       */
      CHAI_HOST_DEVICE ManagedArrayPointer(const ManagedArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }

      /*!
       * @brief Converting copy-constructor from a non-const ManagedArrayPointer to a const ManagedArrayPointer.
       *
       * @tparam OtherElementType The source element type; must be non-const, and this
       *         ManagedArrayPointer's ElementType must be const-qualified version of it.
       *
       * @param other The source ManagedArrayPointer to copy from.
       *
       * @details This constructor is enabled only when converting from
       * ManagedArrayPointer<T, ManagerType> to ManagedArrayPointer<const T, ManagerType>. The manager
       * pointer is shared (ownership semantics are preserved).
       */
      template <typename OtherElementType, 
                typename = std::enable_if_t<!std::is_const_v<OtherElementType> &&
                                            std::is_same_v<ElementType, std::add_const_t<OtherElementType>>>>
      CHAI_HOST_DEVICE ManagedArrayPointer(const ManagedArrayPointer<OtherElementType, ManagerType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
      }

      /*!
       * @brief Copy-assigns from another ManagedArrayPointer.
       *
       * @param other The ManagedArrayPointer to copy from.
       *
       * @return Reference to this ManagedArrayPointer.
       *
       * @details Copies the cached pointer, size, and manager pointer. Ownership
       * semantics are preserved (the manager pointer is shared).
       */
      ManagedArrayPointer& operator=(const ManagedArrayPointer& other) = default;

      /*!
       * @brief Resizes the underlying managed array.
       *
       * @param newSize New number of elements.
       *
       * @details If no manager is associated with this ManagedArrayPointer, a new manager is
       * default-constructed. The cached pointer and size are invalidated (set to nullptr
       * and zero, respectively), and the resize request is forwarded to the manager.
       */
      void resize(std::size_t newSize)
      {
        if (m_manager == nullptr)
        {
          m_manager = new ManagerType();
        }

        m_data = nullptr;
        m_size = 0;
        m_manager->resize(newSize);
      }

      /*!
       * @brief Frees the owned manager and resets this ManagedArrayPointer to null/empty.
       *
       * @details Sets the cached data pointer to nullptr, size to 0, deletes the
       * associated manager (if any), and clears the manager pointer. After calling
       * free(), this ManagedArrayPointer is equivalent to default-constructed.
       */
      void free()
      {
        m_data = nullptr;
        m_size = 0;
        delete m_manager;
        m_manager = nullptr;
      }

      /*!
       * @brief Returns the number of elements in the underlying managed array.
       *
       * @return The number of elements.
       *
       * @details On host builds, synchronizes the cached size from the manager (if present).
       * On device builds (CHAI_DEVICE_COMPILE), returns the last cached size.
       */
      CHAI_HOST_DEVICE std::size_t size() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager)
        {
          m_size = m_manager->size();
        }
#endif
        return m_size;
      }

      /*!
       * @brief Returns the cached pointer to the managed array's data.
       *
       * @return Pointer to the first element of the underlying managed array, or nullptr.
       *
       * @details On host builds, if a manager is present and provides a non-null data pointer,
       * refreshes the cached pointer `m_data` from `m_manager->data()`. On device builds
       * (CHAI_DEVICE_COMPILE), returns the cached pointer without querying the manager.
       */
      CHAI_HOST_DEVICE ElementType* data() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager)
        {
          if (ElementType* data = static_cast<ElementType*>(m_manager->data()); data)
          {
            m_data = data;
          }
        }
#endif
        return m_data;
      }

      /*!
       * @brief Synchronizes the cached pointer and size from the manager.
       *
       * @details On host builds, if a manager is present, refreshes `m_data` from
       * `m_manager->data()` (when non-null) and updates `m_size` from
       * `m_manager->size()`. On device builds (CHAI_DEVICE_COMPILE), this function
       * is a no-op and the cached values are returned as-is.
       */
      CHAI_HOST_DEVICE void update() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager)
        {
          if (ElementType* data = static_cast<ElementType*>(m_manager->data()); data)
          {
            m_data = data;
          }

          m_size = m_manager->size();
        }
#endif
      }

      /*!
       * @brief Unchecked element access.
       *
       * @param i Element index.
       *
       * @return Reference to element i in the cached data pointer.
       *
       * @note No bounds checking is performed.
       * @note Uses the cached pointer `m_data` and does not call update().
       */
      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i) const
      {
        return m_data[i];
      }

    private:
      /*!
       * @brief Cached pointer to the managed array.
       *
       * @details This value is synchronized from the manager by update()/cupdate().
       * It is marked mutable to allow cache refresh in const member functions.
       */
      mutable ElementType* m_data{nullptr};

      /*!
       * @brief Cached number of elements in the managed array.
       *
       * @details This value is synchronized from the manager by size()/update()/cupdate().
       * It is marked mutable to allow cache refresh in const member functions.
       */
      mutable std::size_t m_size{0};

      /*!
       * @brief Pointer to the manager that owns/manages the underlying array.
       *
       * @details Ownership semantics are raw-pointer based: copies share this pointer,
       * and free() may delete it.
       */
      ManagerType* m_manager{nullptr};

      /// Needed for the converting constructor
      template <typename OtherElementType, typename OtherManagerType>
      friend class ManagedArrayPointer;
  };  // class ManagedArrayPointer
}  // namespace chai::expt

#endif  // CHAI_MANAGED_ARRAY_POINTER_HPP
