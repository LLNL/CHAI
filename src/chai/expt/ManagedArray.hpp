#ifndef CHAI_MANAGED_ARRAY_HPP
#define CHAI_MANAGED_ARRAY_HPP

#include "chai/expt/ArrayManager.hpp"
#include "chai/expt/ExecutionContextManager.hpp"
#include <cstddef>

namespace chai {
namespace expt {
  /*!
   * \class ManagedArray
   *
   * \brief An array class that manages coherency across the CPU and GPU.
   *        How the coherence is obtained is controlled by the array manager.
   *
   * \tparam ElementT The type of element in the array.
   */
  template <typename ElementT>
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
      explicit ManagedArray(ArrayManager<ElementT>* manager) :
        m_array_manager{manager}
      {
        if (m_array_manager)
        {
          m_size = m_array_manager->size();
        }
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
        m_data{other.m_data},
        m_size{other.m_size},
        m_array_manager{other.m_array_manager}
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_array_manager) {
          m_data = m_array_manager->data(ExecutionContextManager::getInstance()::getContext(), !std::is_const<ElementT>::value));
        }
#endif
      }

      /*!
       * \brief Sets the array manager for this ManagedArray.
       *
       * \param manager The new array manager to be set.
       *
       * \post The ManagedArray takes ownership of the new manager objet.
       */
       void setManager(ArrayManager<ElementT>* manager)
       {
         delete m_array_manager;
         m_array_manager = manager;
       }

      /*!
       * \brief Get the array manager associated with this ManagedArray.
       *
       * \return A pointer to the array manager.
       */
      ArrayManager<ElementT>* getManager() const {
        return m_array_manager;
      }

      /*!
       * \brief Resizes the array to the specified new size.
       *
       * \param newSize The new size to resize the array to.
       *
       * \note This method updates the size of the array and triggers a resize operation in the array manager if it exists.
       *       If no array manager is associated, an exception is thrown.
       */
      void resize(std::size_t newSize) {
        if (m_array_manager) {
          m_size = newSize;
          m_array_manager->resize(newSize);
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
        delete m_array_manager;
        m_array_manager = nullptr;
      }

      /*!
       * \brief Get the number of elements in the array.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE std::size_t size() const {
        return m_size;
      }

      /*!
       * \brief Get a pointer to the element data in the specified context.
       *
       * \param context The context in which to retrieve the element data.
       *
       * \return A pointer to the element data in the specified context.
       */
      ElementT* data(ExecutionContext context) const {
        if (m_array_manager) {
          m_data = m_array_manager->data(context, !std::is_const<ElementT>::value);
        }

        return m_data;
      }

      /*!
       * \brief Get a const pointer to the element data in the specified context.
       *
       * \param context The context in which to retrieve the const element data.
       *
       * \return A const pointer to the element data in the specified context.
       */
      const ElementT* cdata(ExecutionContext context) const {
        if (m_array_manager) {
          m_data = m_array_manager->data(context, false);
        }

        return m_data;
      }

      /*!
       * \brief Get a pointer to the element data in the current execution space.
       *
       * \return A pointer to the element data in the current execution space.
       */
      CHAI_HOST_DEVICE ElementT* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
        return data(HOST);
#endif
        return m_data;
      }

      /*!
       * \brief Get a const pointer to the element data in the current execution space.
       *
       * \return A const pointer to the element data in the current execution space.
       */
      CHAI_HOST_DEVICE const ElementT* cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
        return cdata(HOST);
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
      CHAI_HOST_DEVICE ElementT& operator[](std::size_t i) const {
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
      ElementT get(std::size_t i) const {
        if (m_array_manager) {
          return m_array_manager->get(i);
        }
        else {
          throw std::runtime_exception("Unable to get element");
        }
      }

      /*!
       * \brief Set a value at a specified index in the array.
       *
       * \param i The index where the value is to be set.
       * \param value The value to set at the specified index.
       *
       * \throw std::runtime_exception if the array manager is not associated with the ManagedArray.
       */
      void set(std::size_t i, const ElementT& value) {
        if (m_array_manager) {
          m_array_manager->set(i, value);
        }
        else {
          throw std::runtime_exception("Unable to set element");
        }
      }

    private:
      /*!
       * The array that is coherent in the current execution space.
       */
      ElementT* m_data = nullptr;

      /*!
       * The number of elements in the array.
       */
      std::size_t m_size = 0;

      /*!
       * The array manager controls the coherence of the array.
       */
      ArrayManager<ElementT>* m_array_manager = nullptr;
  };  // class ManagedArray

  /*!
   * \brief Constructs an array by creating a new manager object.
   *
   * \tparam ArrayManager<ElementT> The type of array manager.
   * \tparam Args The type of the arguments used to construct the array manager.
   *
   * \param args The arguments to construct an array manager.
   */
  template <typename ElementT, typename ArrayManager<ElementT>, typename... Args>
  ManagedArray<ElementT> makeArray(Args&&... args) {
    return ManagedArray<ElementT>(new ArrayManager<ElementT>(std::forward<Args>(args)...));
  }
}  // namespace expt
}  // namespace chai

#endif  // CHAI_MANAGED_ARRAY_HPP
