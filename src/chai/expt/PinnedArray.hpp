#ifndef CHAI_PINNED_ARRAY_HPP
#define CHAI_PINNED_ARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
  /*!
   * Represents an array backed by pinned memory. If used with RAJA, CHAI
   * will take care of synchronizations needed when switching between using
   * the array on the host and device.
   */
  template <class T>
  class PinnedArray {
    public:
      /*!
       * \brief Default constructor
       */
      PinnedArray() = default;

      /*!
       * \brief Default constructor
       *
       * \param count The new number of elements
       * \param allocator_id The ID of the Umpire allocator to use
       */
      PinnedArray(
        size_t count,
        int allocator_id = getDefaultPinnedAllocatorId())
        : m_allocator_id{allocator_id}
      {
        resize(count);
      }

      /*!
       * \brief Copy constructor
       *
       * \note This object is intended to be passed around by value,
       *       so the copy constructor is intentionally shallow.
       */
      PinnedArray(const PinnedArray& other) = default;

      /*!
       * \brief Aliasing constructor. Allows converting the element type to const.
       *
       * \param other PinnedArray with non-const element type
       */
      template <class U, class std::enable_if<!std::is_const<U>::value, int>::type = 0>
      PinnedArray(const PinnedArray<U>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_allocator_id{other.m_allocator_id}
      {
      }

      /*
       * \brief Assignment operator
       *
       * \note This object is intended to be passed around by value,
       *       so the assignment operator is intentionally shallow.
       */
      PinnedArray& operator=(const PinnedArray& other) = default;

      /*!
       * \brief Resizes the pinned array
       *
       * \param count The new number of elements
       */
      void resize(size_type count)
      {
        if (count == 0)
        {
          free();
        }
        else if (m_size == 0)
        {
          m_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).allocate(count * sizeof(T)));
        }
        else if (m_size != count)
        {
          size_type bytesToCopy = m_size < count : m_size * sizeof(T)
                                                 ? count * sizeof(T);

          T* newData = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).allocate(count * sizeof(T)));

          umpire::ResourceManager::getInstance().copy(newData, m_data, bytesToCopy);
          // TODO: Default initialize? It could be done on the host.

          umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).deallocate(m_data);
          m_data = newData;
        }

        m_size = count;
      }

      /*!
       * \brief Frees the pinned array
       */
      void free()
      {
        umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).deallocate(m_data);
        m_data = nullptr;
        m_size = 0;
      }

      /*!
       * \brief Get the number of elements in the array
       *
       * \return The number of elements in the array
       */
      CHAI_HOST_DEVICE size_type size() const
      {
        return m_size;
      }

     /*!
      * \brief Return reference to i-th element of the PinnedArray.
      *
      * \param i Element to return reference to.
      *
      * \return Reference to i-th element.
      */
     CHAI_HOST_DEVICE T& operator[](size_type i) const
     {
       return m_data[i];
     }

     /*!
      * \brief Return a pointer that can be used in the given execution space
      *
      * \param space The execution space in which the returned pointer will be used
      * \return Pointer to data in the given execution space
      */
     T* data(ExecutionSpace space) const
     {
       if (space == CPU)
       {
         ArrayManager::getInstance()->syncIfNeeded();
       }

       return m_data;
     }

     /*!
      * \brief Return a pointer that can be used in the current execution space
      *
      * \return Pointer to data in the current execution space
      */
     CHAI_HOST_DEVICE T* data() const
     {
#if defined(CHAI_DEVICE_COMPILE)
       return m_data;
#else
       return data(CPU);
#endif
     }

     /*!
      * \brief Return a pointer that can be used in the given execution space
      *
      * \param space The execution space in which the returned pointer will be used
      * \return Pointer to data in the given execution space
      *
      * \note Do not mark data as touched in the requested execution space
      */
     const T* cdata(ExecutionSpace space) const
     {
       return data(space);
     }

     /*!
      * \brief Return a pointer that can be used in the current execution space
      *
      * \return Pointer to data in the current execution space
      *
      * \note Do not mark data as touched in the current execution space
      */
     CHAI_HOST_DEVICE const T* cdata() const
     {
       return data();
     }

     /*!
      * \brief Compare one PinnedArray with another for equality
      *
      * \param other The other PinnedArray
      *
      * \return true if all members are equal
      */
     CHAI_HOST_DEVICE bool operator==(const PinnedArray<T>& other) const
     {
       return m_data == other.m_data &&
              m_size == other.m_size &&
              m_allocator_id == other.m_allocator_id;
     }

     /*!
      * \brief Compare one PinnedArray with another for inequality
      *
      * \param other The other PinnedArray
      *
      * \return true if not all members are equal
      */
     CHAI_HOST_DEVICE bool operator!=(const PinnedArray<T>& other) const
     {
       return !(this == other);
     }

     /*!
      * \brief Conversion to bool operator
      *
      * \return true if the PinnedArray is not empty
      */
     CHAI_HOST_DEVICE explicit operator bool() const
     {
       return m_size > 0;
     }

#if defined(CHAI_ENABLE_PICK)
     /*!
      * \brief Return the value of element i in the PinnedArray.
      *
      * \param index The index of the element to be fetched
      * \return The value of the i-th element in the PinnedArray.
      */
     CHAI_HOST_DEVICE T pick(size_type index) const
     {
#if !defined(CHAI_DEVICE_COMPILE)
       ArrayManager::getInstance()->syncIfNeeded();
#endif
       return m_data[index];
     }

     /*!
      * \brief Set the value of element i in the PinnedArray to be value.
      *
      * \param index The index of the element to be set
      * \param value Source location of the value
      */
     CHAI_HOST_DEVICE void set(size_type index, T value) const
     {
#if !defined(CHAI_DEVICE_COMPILE)
       ArrayManager::getInstance()->syncIfNeeded();
#endif
       m_data[index] = value;
     }

   private:
     mutable T* m_data = nullptr;
     mutable size_type m_size = 0;
     int m_allocator_id = getDefaultPinnedAllocatorId();
  };  // class PinnedArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_HPP
