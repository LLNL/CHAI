#ifndef CHAI_MANAGED_ARRAY_HPP
#define CHAI_MANAGED_ARRAY_HPP

#include "chai/expt/ArrayManager.hpp"
#include <cstddef>

namespace chai {
namespace expt {
  template <typename ElementType, typename ArrayType>
  class Array {
    public:
      Array() = default;

      explicit Array(const ArrayType& manager)
        : m_manager{manager}
      {
      }

      explicit Array(ArrayType&& manager)
        : m_manager{std::move(manager)}
      {
      }

      Array(const Array& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_manager{other.m_manager}
      {
        update();
      }
      
      void resize(std::size_t newSize) {
        m_data = nullptr;
        m_size = newSize;
        m_manager.resize(newSize);
      }

      void free() {
        m_data = nullptr;
        m_size = 0;
        m_manager.free();
      }

      CHAI_HOST_DEVICE std::size_t size() const
      {
        return m_size;
      }

      CHAI_HOST_DEVICE void update() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        m_data = m_manager.data(!std::is_const_v<ElementType>);
#endif
      }

      CHAI_HOST_DEVICE void cupdate() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        m_data = m_manager.data(false);
#endif
      }

      /*!
       * \brief Get a pointer to the element data in the specified context.
       *
       * \param context The context in which to retrieve the element data.
       *
       * \return A pointer to the element data in the specified context.
       */
      ElementType* data() const
      {
        update();
        return m_data;
      }

      /*!
       * \brief Get a const pointer to the element data in the specified context.
       *
       * \param context The context in which to retrieve the const element data.
       *
       * \return A const pointer to the element data in the specified context.
       */
      CHAI_HOST_DEVICE const ElementType* cdata() const {
        cupdate();
        return m_data;
      }

      /*!
       * \brief Get the ith element in the array.
       *
       * \param i The index of the element to retrieve.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i) const
      {
        return m_data[i];
      }

      /*!
       * \brief Get the value of the element at the specified index.
       *
       * \param i The index of the element to retrieve.
       *
       * \return The value of the element at the specified index.
       *
       * \throw std::runtime_exception if unable to retrieve the element.
       */
      ElementType get(std::size_t i) const
      {
        return *(static_cast<ElementType*>(m_manager.get(i*sizeof(ElementType), sizeof(ElementType))));
      }

      /*!
       * \brief Set a value at a specified index in the array.
       *
       * \param i The index where the value is to be set.
       * \param value The value to set at the specified index.
       *
       * \throw std::runtime_exception if the array manager is not associated with the Array.
       */
      void set(std::size_t i, const ElementType& value) {
        m_manager.set(i*sizeof(ElementType), sizeof(ElementType), static_cast<void*>(std::addressof(value)));
      }

    private:
      /*!
       * The array that is coherent in the current execution space.
       */
      ElementType* m_data{nullptr};

      /*!
       * The number of elements in the array.
       */
      std::size_t m_size{0};

      /*!
       * The array manager controls the coherence of the array.
       */
      ArrayType m_manager{};
  };  // class Array

  /*!
   * \brief Constructs an array by creating a new manager object.
   *
   * \tparam ArrayManager<ElementType> The type of array manager.
   * \tparam Args The type of the arguments used to construct the array manager.
   *
   * \param args The arguments to construct an array manager.
   */
  template <typename ElementType, typename ArrayManager, typename... Args>
  Array<ElementType> makeArray(Args&&... args) {
    return Array<ElementType>(new ArrayManager(std::forward<Args>(args)...));
  }
}  // namespace expt
}  // namespace chai

#endif  // CHAI_MANAGED_ARRAY_HPP
