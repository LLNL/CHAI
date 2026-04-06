//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXPT_MANAGED_ARRAY_VIEW_HPP
#define CHAI_EXPT_MANAGED_ARRAY_VIEW_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"

#include <cstddef>
#include <type_traits>
namespace chai::expt
{
  /*!
   * \brief This class provides a uniform interface for viewing different types
   *        of managed array memory across multiple backends.
   *
   * \details This class has shallow-copy semantics, which allows it to be
   *          passed by value to a CUDA or HIP kernel. When copy constructed,
   *          it queries the array manager to update the cached size and pointer
   *          from the array manager. This acts as a non-owning view and cannot
   *          allocate, free, or resize the underlying data.
   *
   * \tparam ElementType The type of elements contained in this array.
   * \tparam ManagerType Manages the underlying memory.
   */
  template <typename ElementType, typename ManagerType>
  class ManagedArrayView
  {
    public:
      /*!
       * \brief Constructs a default ManagedArrayView.
       *
       * \details Creates a null pointer with size zero and no associated manager.
       */
      ManagedArrayView() = default;

      /*!
       * \brief Constructs a ManagedArrayView from an existing manager.
       *
       * \param manager An object that owns/manages the underlying array.
       *
       * \note The manager is not copied or owned. The caller must ensure the
       *       manager outlives the view.
       */
      explicit ManagedArrayView(ManagerType& manager)
        : m_data{manager.data()},
          m_size{manager.size()},
          m_manager{&manager}
      {
      }

      /*!
       * \brief Copy-constructs a ManagedArrayView from another ManagedArrayView.
       *
       * \param other The ManagedArrayView to copy.
       *
       * \details Copies the cached pointer, size, and manager pointer. The
       * cached pointer/size are synchronized by calling update().
       */
      CHAI_HOST_DEVICE ManagedArrayView(const ManagedArrayView& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }

      /*!
       * \brief Converting copy-constructor from a non-const ManagedArrayView to a const ManagedArrayView.
       *
       * \tparam OtherElementType The source element type; must be non-const, and this
       *         ManagedArrayView's ElementType must be const-qualified version of it.
       *
       * \param other The source ManagedArrayView to copy from.
       */
      template <typename OtherElementType,
                typename = std::enable_if_t<!std::is_const_v<OtherElementType> &&
                                            std::is_same_v<ElementType, std::add_const_t<OtherElementType>>>>
      CHAI_HOST_DEVICE ManagedArrayView(const ManagedArrayView<OtherElementType, ManagerType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
      }

      /*!
       * \brief Copy-assigns from another ManagedArrayView.
       *
       * \param other The ManagedArrayView to copy from.
       *
       * \return Reference to this ManagedArrayView.
       */
      ManagedArrayView& operator=(const ManagedArrayView& other) = default;

      /*!
       * \brief Returns the number of elements in the underlying managed array.
       *
       * \return The number of elements.
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
       * \brief Returns the cached pointer to the managed array's data.
       *
       * \return Pointer to the first element of the underlying managed array, or nullptr.
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
       * \brief Synchronizes the cached pointer and size from the manager.
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
       * \brief Unchecked element access.
       *
       * \param i Element index.
       *
       * \return Reference to element i in the cached data pointer.
       */
      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i) const
      {
        return m_data[i];
      }

    private:
      /*!
       * \brief Cached pointer to the managed array.
       */
      mutable ElementType* m_data{nullptr};

      /*!
       * \brief Cached number of elements in the managed array.
       */
      mutable std::size_t m_size{0};

      /*!
       * \brief Pointer to the manager that owns/manages the underlying array.
       */
      ManagerType* m_manager{nullptr};

      /// Needed for the converviting constructor
      template <typename OtherElementType, typename OtherManagerType>
      friend class ManagedArrayView;
  };  // class ManagedArrayView
}  // namespace chai::expt

#endif  // CHAI_EXPT_MANAGED_ARRAY_VIEW_HPP
