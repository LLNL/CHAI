#ifndef CHAI_MANAGED_ARRAY_HPP
#define CHAI_MANAGED_ARRAY_HPP

#include "chai/ArrayManager.hpp"

namespace chai {
  /*!
   * \class ManagedArray
   *
   * \brief An array class that manages coherency across the CPU and GPU.
   *        How the coherence is obtained is controlled by the array manager.
   *
   * \tparam T The type of element in the array.
   */
  template <typename T>
  class ManagedArray {
    public:
      /*!
       * \brief Constructs an empty array without an array manager.
       */
      ManagedArray() = default;

      /*!
       * \brief Constructs an array from a manager.
       *
       * \param manager The array manager controls the coherence of the array.
       *
       * \note The array takes ownership of the manager.
       */
      ManagedArray(ArrayManager* manager) :
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
      CHAI_HOST_DEVICE ManagedArray(const ManagedArray& other) :
        m_size{other.m_size},
        m_data{other.m_data},
        m_manager{other.m_manager}
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_manager->update(m_size, m_data);
        }
      }

      /*!
       * \brief Frees the resources associated with this array.
       *
       * \note Once free has been called, it is invalid to use any other copies
       *       of this array (since copies are shallow).
       */
      void free() {
        m_size = 0;
        m_data = nullptr;
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
       * The number of elements in the array.
       */
      size_t m_size = 0;

      /*!
       * The array that is coherent in the current execution space.
       */
      T* m_data = nullptr;

      /*!
       * The array manager controls the coherence of the array.
       */
      ArrayManager* m_manager = nullptr;
  };  // class ManagedArray

  /*!
   * \brief Constructs an array by creating a new manager object.
   *
   * \tparam Manager The type of array manager.
   * \tparam Args The type of the arguments used to construct the array manager.
   *
   * \param args The arguments to construct an array manager.
   */
  template <typename Manager, typename... Args>
  ManagedArray<T> makeManagedArray(Args&&... args) {
    return ManagedArray<T>(new Manager(std::forward<Args>(args)...));
  }
}  // namespace chai

#endif  // CHAI_MANAGED_ARRAY_HPP
