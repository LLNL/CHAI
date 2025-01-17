#ifndef CHAI_DEVICE_ARRAY_HPP
#define CHAI_DEVICE_ARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
  /*!
   * Represents an array backed by host memory.
   */
  template <class T>
  class DeviceArray {
    public:
      /*!
       * \brief Default constructor
       */
      DeviceArray() = default;

      /*!
       * \brief Default constructor
       *
       * \param count The new number of elements
       * \param allocator_id The ID of the Umpire allocator to use
       */
      DeviceArray(
        size_type count,
        int allocator_id = getDefaultDeviceAllocatorId())
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
      DeviceArray(const DeviceArray& other) = default;

      /*!
       * \brief Aliasing constructor. Allows converting the element type to const.
       *
       * \param other DeviceArray with non-const element type
       */
      template <class U, class std::enable_if<!std::is_const<U>::value, int>::type = 0>
      CHAI_HOST_DEVICE DeviceArray(const DeviceArray<U>& other)
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
      DeviceArray& operator=(const DeviceArray& other) = default;

      /*!
       * \brief Resizes the device array
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
       * \brief Frees the device array
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
      CHAI_DEVICE size_type size() const
      {
        return m_size;
      }

     /*!
      * \brief Return reference to i-th element of the DeviceArray.
      *
      * \param i Element to return reference to.
      *
      * \return Reference to i-th element.
      */
     CHAI_DEVICE T& operator[](size_type i) const
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
       if (space == GPU)
       {
         return m_data;
       }
       else
       {
         return nullptr;
       }
     }

     /*!
      * \brief Return a pointer that can be used in the current execution space
      *
      * \return Pointer to data in the current execution space
      */
     CHAI_DEVICE T* data() const
     {
       return m_data;
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
     CHAI_DEVICE const T* cdata() const
     {
       return data();
     }

     /*!
      * \brief Compare one DeviceArray with another for equality
      *
      * \param other The other DeviceArray
      *
      * \return true if all members are equal
      */
     CHAI_DEVICE bool operator==(const DeviceArray<T>& other) const
     {
       return m_data == other.m_data &&
              m_size == other.m_size &&
              m_allocator_id == other.m_allocator_id;
     }

     /*!
      * \brief Compare one DeviceArray with another for inequality
      *
      * \param other The other DeviceArray
      *
      * \return true if not all members are equal
      */
     CHAI_DEVICE bool operator!=(const DeviceArray<T>& other) const
     {
       return !(this == other);
     }

     /*!
      * \brief Conversion to bool operator
      *
      * \return true if the DeviceArray is not empty
      */
     CHAI_DEVICE explicit operator bool() const
     {
       return m_size > 0;
     }

#if defined(CHAI_ENABLE_PICK)
     /*!
      * \brief Return the value of element i in the DeviceArray.
      *
      * \param index The index of the element to be fetched
      * \return The value of the i-th element in the DeviceArray.
      */
     CHAI_DEVICE T pick(size_type index) const
     {
       return m_data[index];
     }

     /*!
      * \brief Set the value of element i in the DeviceArray to be value.
      *
      * \param index The index of the element to be set
      * \param value Source location of the value
      */
     CHAI_DEVICE void set(size_type index, T value) const
     {
       m_data[index] = value;
     }

   private:
     mutable T* m_data = nullptr;
     mutable size_type m_size = 0;
     int m_allocator_id = getDefaultDeviceAllocatorId();
  };  // class DeviceArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_DEVICE_ARRAY_HPP
