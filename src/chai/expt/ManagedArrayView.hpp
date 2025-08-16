#ifndef CHAI_ARRAY_VIEW_HPP
#define CHAI_ARRAY_VIEW_HPP

#include "chai/Manager.hpp"
#include <cstddef>

namespace chai {
namespace expt {
  /*!
   * \class ArrayView
   *
   * \brief A view into an existing Array without taking ownership of the data.
   *
   * \tparam T The type of element in the array view.
   */
  template <typename T>
  class ArrayView {
    public:
      /*!
       * \brief Constructs an empty array view.
       */
      ArrayView() = default;

      /*!
       * \brief Constructs an array view from a manager.
       *
       * \param manager The array manager controls the coherence of the array.
       */
      explicit ArrayView(Manager* manager) :
        m_manager{manager}
      {
        if (m_manager)
        {
          m_size = m_manager->size();
        }
      }

      /*!
       * \brief Constructs an array view with specified size and manager.
       *
       * \param size The number of elements
       * \param manager The array manager
       */
      ArrayView(std::size_t offset, std::size_t size, Manager* manager) :
        m_offset{offset},  
        m_size{size},
        m_manager{manager}
      {
      }

      /*!
       * \brief Constructs a shallow copy of an array view from another and makes
       *        the data coherent in the current execution space.
       *
       * \param other The other array view.
       *
       * \note This is a shallow copy.
       */
      CHAI_HOST_DEVICE ArrayView(const ArrayView& other) :
        m_data{other.m_data},
        m_offset{other.m_offset},
        m_size{other.m_size},
        m_manager{other.m_manager}
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(ContextManager::getInstance()::getContext(), !std::is_const<T>::value)) + m_offset;
        }
#endif
      }

      CHAI_HOST_DEVICE std::size_t offset() const {
        return m_offset;
      }

      /*!
       * \brief Get the number of elements in the array view.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE std::size_t size() const {
        return m_size;
      }

      CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(ExecutionContext::HOST, !std::is_const<T>::value)) + m_offset;
        }
#endif
        return m_data;
      }

      CHAI_HOST_DEVICE T* data(ExecutionContext context) const {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(context, !std::is_const<T>::value)) + m_offset;
        }
#endif
        return m_data;
      }

      /*!
       * \brief Get the ith element in the array view.
       *
       * \param i The index of the element to retrieve.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE T& operator[](std::size_t i) const {
        return m_data[i];
      }

    private:
      /*!
       * The array that is coherent in the current execution space.
       */
      T* m_data = nullptr;

      /*!
       * The starting element in the array view.
       */
       std::size_t m_offset = 0;

      /*!
       * The number of elements in the array view.
       */
      std::size_t m_size = 0;

      /*!
       * The array manager controls the coherence of the array.
       * ArrayView doesn't own the manager
       */
      Manager* m_manager = nullptr;
  };  // class ArrayView

  /*!
   * \brief Constructs an array view by using an existing manager object.
   *
   * \tparam Manager The type of array manager.
   */
  template <typename T, typename Manager>
  ArrayView<T> makeArrayView(Manager* manager) {
    return ArrayView<T>(manager);
  }
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_VIEW_HPP