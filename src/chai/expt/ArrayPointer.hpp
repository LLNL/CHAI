//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_ARRAY_POINTER_HPP
#define CHAI_ARRAY_POINTER_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include <cstddef>
#include <type_traits>

namespace chai::expt
{
  template <typename ElementType, typename ManagerType>
  class ArrayPointer
  {
    public:
      /*!
       * @brief Constructs a default ArrayPointer.
       *
       * @details Creates a null pointer with size 0 and no associated manager.
       */
      ArrayPointer() = default;

      /*!
       * @brief Constructs an ArrayPointer from an existing manager.
       *
       * @param manager Pointer to a manager that owns/manages the underlying array.
       *
       * @details The ArrayPointer assumes pointer ownership semantics, meaning
       * this ArrayPointer or any copy of this ArrayPointer can delete the manager.
       */
      explicit ArrayPointer(Manager* manager)
        : m_manager{manager}
      {
      }

      /*!
       * @brief Copy-constructs an ArrayPointer from another ArrayPointer.
       *
       * @param other The ArrayPointer to copy.
       *
       * @details Copies the cached pointer, size, and manager pointer. Ownership
       * semantics are preserved (the manager pointer is shared). The internal
       * cached pointer/size are synchronized by calling update().
       */
      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }

      /*!
       * @brief Converting copy-constructor from a non-const ArrayPointer to a const ArrayPointer.
       *
       * @tparam OtherElementType The source element type; must be non-const, and this
       *         ArrayPointer's ElementType must be const-qualified version of it.
       *
       * @param other The source ArrayPointer to copy from.
       *
       * @details This constructor is enabled only when converting from
       * ArrayPointer<T, ManagerType> to ArrayPointer<const T, ManagerType>. The manager
       * pointer is shared (ownership semantics are preserved).
       */
      template <typename OtherElementType, 
                typename = std::enable_if_t<!std::is_const_v<OtherElementType> &&
                                            std::is_same_v<ElementType, std::add_const_t<OtherElementType>>>>
      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer<OtherElementType, ManagerType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
      }

      /*!
       * @brief Copy-assigns from another ArrayPointer.
       *
       * @param other The ArrayPointer to copy from.
       *
       * @return Reference to this ArrayPointer.
       *
       * @details Copies the cached pointer, size, and manager pointer. Ownership
       * semantics are preserved (the manager pointer is shared).
       */
      ArrayPointer& operator=(const ArrayPointer& other) = default;

      /*!
       * @brief Resizes the underlying managed array.
       *
       * @param newSize New number of elements.
       *
       * @details If no manager is associated with this ArrayPointer, a new manager is
       * default-constructed. The cached pointer and size are invalidated (set to nullptr
       * and zero, respectively), and the resize request is forwarded to the manager.
       */
      void resize(std::size_t newSize)
      {
        if (m_manager == nullptr)
        {
          m_manager = new Manager();
        }

        m_data = nullptr;
        m_size = 0;
        m_manager->resize(newSize);
      }

      /*!
       * @brief Frees the owned manager and resets this ArrayPointer to null/empty.
       *
       * @details Sets the cached data pointer to nullptr, size to 0, deletes the
       * associated manager (if any), and clears the manager pointer. After calling
       * free(), this ArrayPointer is equivalent to default-constructed.
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
      Manager* m_manager{nullptr};

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
          if (ElementType* data = m_manager->data(); data)
          {
            m_data = data;
          }

          m_size = m_manager->size();
        }
#endif
      }

      /// Needed for the converting constructor
      template <typename OtherElementType, template <typename> typename OtherManagerType>
      friend class ArrayPointer;
  };  // class ArrayPointer
}  // namespace chai::expt

#endif  // CHAI_ARRAY_POINTER_HPP
