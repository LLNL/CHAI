//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_ARRAY_VIEW_HPP
#define CHAI_ARRAY_VIEW_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include <cstddef>
#include <type_traits>

namespace chai::expt
{
  template <typename ElementType, typename ManagerType>
  class ArrayView
  {
    public:
      /*!
       * @brief Constructs a default ArrayView.
       *
       * @details Creates a null pointer with size 0 and no associated manager.
       */
      ArrayView() = default;

      /*!
       * @brief Constructs an ArrayView from an existing manager.
       *
       * @param manager Pointer to a manager that owns/manages the underlying array.
       *
       * @details The ArrayView does not take ownership of the manager.
       */
      explicit ArrayView(Manager* manager)
        : m_manager{manager}
      {
      }

      /*!
       * @brief Copy-constructs an ArrayView from another ArrayView.
       *
       * @param other The ArrayView to copy.
       *
       * @details Performs a shallow copy of the cached pointer, size, and manager pointer.
       * The internal cached pointer/size are synchronized by calling update().
       */
      CHAI_HOST_DEVICE ArrayView(const ArrayView& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }

      /*!
       * @brief Converting copy-constructor from a non-const ArrayView to a const ArrayView.
       *
       * @tparam OtherElementType The source element type; must be non-const, and this
       *         ArrayView's ElementType must be const-qualified version of it.
       *
       * @param other The source ArrayView to copy from.
       *
       * @details This constructor is enabled only when converting from
       * ArrayView<T, ManagerType> to ArrayView<const T, ManagerType>.
       * Performs a shallow copy of the cached pointer, size, and manager pointer.
       */
      template <typename OtherElementType, 
                typename = std::enable_if_t<!std::is_const_v<OtherElementType> &&
                                            std::is_same_v<ElementType, std::add_const_t<OtherElementType>>>>
      CHAI_HOST_DEVICE ArrayView(const ArrayView<OtherElementType, ManagerType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
      }

      /*!
       * @brief Copy-assigns from another ArrayView.
       *
       * @param other The ArrayView to copy from.
       *
       * @return Reference to this ArrayView.
       *
       * @details Performs a shallow copy of the cached pointer, size, and manager pointer.
       */
      ArrayView& operator=(const ArrayView& other) = default;

      /*!
       * @brief Returns the number of elements in the underlying managed array.
       *
       * @return The number of elements.
       *
       * @details If called on the host, synchronizes the cached size from the manager (if present).
       * On the device, returns the last cached size.
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
       * @note Uses the cached pointer `m_data` and does not call update()
       * to ensure data coherence.
       */
      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i) const
      {
        return m_data[i];
      }

    private:
      /*!
       * @brief Cached pointer to the managed array.
       *
       * @details This value is synchronized from the manager by update().
       * It is marked mutable to allow cache refresh in const member functions.
       */
      mutable ElementType* m_data{nullptr};

      /*!
       * @brief Cached number of elements in the managed array.
       *
       * @details This value is synchronized from the manager by size()/update().
       * It is marked mutable to allow cache refresh in const member functions.
       */
      mutable std::size_t m_size{0};

      /*!
       * @brief Pointer to the manager that owns/manages the underlying array.
       *
       * @details The ArrayView does not own the manager.
       */
      Manager* m_manager{nullptr};

      /*!
       * @brief Synchronizes the cached pointer and size from the manager.
       *
       * @details If called on the host, synchronizes the cached data and size from the manager (if present).
       * On the device, this function is a no-op and the cached values are returned as-is.
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
  };  // class ArrayView
}  // namespace chai::expt

#endif  // CHAI_ARRAY_VIEW_HPP
