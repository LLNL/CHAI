#ifndef CHAI_ARRAY_HPP
#define CHAI_ARRAY_HPP

#include "chai/Manager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class Array
   *
   * \brief An array class that manages coherency across the CPU and GPU.
   *        How the coherence is obtained is controlled by the array manager.
   *
   * \tparam T The type of element in the array.
   */
  template <typename T>
  class Array {
    public:
      /*!
       * \brief Constructs an empty array without an array manager.
       */
      Array() = default;

      /*!
       * \brief Constructs an array from a manager.
       *
       * \param manager The array manager controls the coherence of the array.
       *
       * \note The array takes ownership of the manager.
       */
      explicit Array(Manager* manager) :
        m_manager{manager}
      {
      }

      /*!
       * \brief Constructs a shallow copy of an array from another and makes
       *        the data coherent in the current execution space.
       *
       * \param other The other array.
       *
       * \note This is a shallow copy.
       */
      CHAI_HOST_DEVICE Array(const Array& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_manager{other.m_manager}
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(!std::is_const<T>::value));
        }
#endif
      }

      void setManager(Manager* manager)
      {
        delete m_manager;
        m_manager = manager;
      }

      void resize(size_t newSize) {
        if (m_manager) {
          m_size = newSize;
          m_manager->resize(newSize);
        }
        else {
          throw std::runtime_exception("Unable to resize");
        }
      }

      /*!
       * \brief Frees the resources associated with this array.
       *
       * \note Once free has been called, it is invalid to use any other copies
       *       of this array (since copies are shallow).
       */
      void free() {
        m_data = nullptr;
        m_size = 0;
        delete m_manager;
        m_manager = nullptr;
      }


      /*!
       * \brief Get the number of elements in the array.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE size_t size() const {
        return m_size;
      }

      CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(!std::is_const<T>::value));
        }
#endif
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
      CHAI_HOST_DEVICE T& operator[](size_t i) const {
        return m_data[i];
      }

    private:
      /*!
       * The array that is coherent in the current execution space.
       */
      T* m_data = nullptr;

      /*!
       * The number of elements in the array.
       */
      size_t m_size = 0;

      /*!
       * The array manager controls the coherence of the array.
       */
      Manager* m_manager = nullptr;
  };  // class Array

  /*!
   * \brief Constructs an array by creating a new manager object.
   *
   * \tparam Manager The type of array manager.
   * \tparam Args The type of the arguments used to construct the array manager.
   *
   * \param args The arguments to construct an array manager.
   */
  template <typename T, typename Manager, typename... Args>
  Array<T> makeArray(Args&&... args) {
    return Array<T>(new Manager(std::forward<Args>(args)...));
  }
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_HPP
