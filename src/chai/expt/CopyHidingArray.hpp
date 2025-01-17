#ifndef CHAI_COPY_HIDING_ARRAY_HPP
#define CHAI_COPY_HIDING_ARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
  /*!
   * Represents an array backed by copy hiding memory. If used with RAJA, CHAI
   * will take care of synchronizations needed when switching between using
   * the array on the host and device.
   */
  template <class T>
  class CopyHidingArray {
    public:
      /*!
       * \brief Default constructor
       */
      CopyHidingArray() = default;

      /*!
       * \brief Default constructor
       *
       * \param count The new number of elements
       * \param allocator_id The ID of the Umpire allocator to use
       */
      CopyHidingArray(
        size_t count,
        int host_allocator_id = getDefaultHostAllocatorID(),
        int device_allocator_id = getDefaultDeviceAllocatorID())
        : m_host_allocator_id{host_allocator_id},
          m_device_allocator_id{device_allocator_id}
      {
        resize(count);
      }

      /*!
       * \brief Copy constructor
       *
       * \note This object is intended to be passed around by value,
       *       so the copy constructor is intentionally shallow.
       */
      CopyHidingArray(const CopyHidingArray& other) = default;

      /*!
       * \brief Aliasing constructor. Allows converting the element type to const.
       *
       * \param other CopyHidingArray with non-const element type
       */
      template <class U, class std::enable_if<!std::is_const<U>::value, int>::type = 0>
      CopyHidingArray(const CopyHidingArray<U>& other)
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
      CopyHidingArray& operator=(const CopyHidingArray& other) = default;

      /*!
       * \brief Resizes the copy hiding array
       *
       * \param count The new number of elements
       */
      void resize(size_type count, ExecutionSpace space = getDefaultAllocationSpace())
      {
        if (count == 0)
        {
          free();
        }
        else if (m_size == 0)
        {
          if (space == CPU)
          {
            m_host_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(count * sizeof(T)));
          }
          else if (space == GPU)
          {
            m_device_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(count * sizeof(T)));
          }
        }
        else if (m_size != count)
        {
          size_type bytesToCopy = m_size < count : m_size * sizeof(T)
                                                 ? count * sizeof(T);

          if (m_last_touch == CPU) {
            T* newData = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(count * sizeof(T)));

            umpire::ResourceManager::getInstance().copy(newData, m_host_data, bytesToCopy);
            // TODO: Default initialize? It could be done on the host.

            umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
            m_host_data = newData;

            umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
            m_device_data = nullptr;
          }
          else if (m_last_touch == GPU)
          {
            T* newData = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(count * sizeof(T)));

            umpire::ResourceManager::getInstance().copy(newData, m_device_data, bytesToCopy);
            // TODO: Default initialize? It could be done on the device.

            umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
            m_device_data = newData;

            umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
            m_host_data = nullptr;
          }
          else
          {
            // This could mean two things. Either neither array is initialized,
            // or both are initialized and up to date.
            T* newData = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(count * sizeof(T)));

            umpire::ResourceManager::getInstance().copy(newData, m_host_data, bytesToCopy);
            // TODO: Default initialize? It could be done on the host.

            umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
            m_host_data = newData;

            T* newData = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(count * sizeof(T)));

            umpire::ResourceManager::getInstance().copy(newData, m_device_data, bytesToCopy);
            // TODO: Default initialize? It could be done on the device.

            umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
            m_device_data = newData;
          }
        }

        m_size = count;
      }

      /*!
       * \brief Frees the copy hiding array
       */
      void free()
      {
        umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
        m_host_data = nullptr;

        umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
        m_device_data = nullptr;

        m_size = 0;
      }

      /*!
       * \brief Frees the copy hiding array
       */
      void free(ExecutionSpace space)
      {
        if (space == CPU) {
          if (m_last_touch == CPU || m_last_touch == NONE) {
            if (m_device_data == nullptr) {
              m_device_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(m_size * sizeof(T)));
            }

            umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
            // TODO: Default initialize? It could be done on the host.
          }

          umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
          m_host_data = nullptr;
        }
        else if (space == GPU) {
          if (m_last_touch == GPU || m_last_touch == NONE) {
            if (m_host_data == nullptr) {
              m_host_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(m_size * sizeof(T)));
            }

            umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
            // TODO: Default initialize? It could be done on the host.
          }

          umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
          m_device_data = nullptr;
        }
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
      * \brief Return reference to i-th element of the CopyHidingArray.
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
         if (m_host_data == nullptr) {
           m_host_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(m_size * sizeof(T)));
         }

         if (m_last_touched == GPU)
         {
           umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
           m_last_touched = NONE;
         }

         if (!std::is_const<T>::value)
         {
           m_last_touched = CPU;
         }

         return m_host_data;
       }
       else if (space == GPU)
       {
         if (m_device_data == nullptr) {
           m_device_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(m_size * sizeof(T)));
         }

         if (m_last_touched == CPU)
         {
           umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
           m_last_touched = NONE;
         }

         if (!std::is_const<T>::value)
         {
           m_last_touched = GPU;
         }

         return m_device_data;
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
       if (space == CPU)
       {
         if (m_host_data == nullptr) {
           m_host_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(m_size * sizeof(T)));
         }

         if (m_last_touched == GPU)
         {
           umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
           m_last_touched = NONE;
         }

         return m_host_data;
       }
       else if (space == GPU)
       {
         if (m_device_data == nullptr) {
           m_device_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(m_size * sizeof(T)));
         }

         if (m_last_touched == CPU)
         {
           umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
           m_last_touched = NONE;
         }

         return m_device_data;
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
      *
      * \note Do not mark data as touched in the current execution space
      */
     CHAI_HOST_DEVICE const T* cdata() const
     {
#if defined(CHAI_DEVICE_COMPILE)
       return m_data;
#else
       return cdata(CPU);
#endif
     }

     /*!
      * \brief Compare one CopyHidingArray with another for equality
      *
      * \param other The other CopyHidingArray
      *
      * \return true if all members are equal
      */
     CHAI_HOST_DEVICE bool operator==(const CopyHidingArray<T>& other) const
     {
       return m_data == other.m_data &&
              m_size == other.m_size &&
              m_allocator_id == other.m_allocator_id;
     }

     /*!
      * \brief Compare one CopyHidingArray with another for inequality
      *
      * \param other The other CopyHidingArray
      *
      * \return true if not all members are equal
      */
     CHAI_HOST_DEVICE bool operator!=(const CopyHidingArray<T>& other) const
     {
       return !(this == other);
     }

     /*!
      * \brief Conversion to bool operator
      *
      * \return true if the CopyHidingArray is not empty
      */
     CHAI_HOST_DEVICE explicit operator bool() const
     {
       return m_size > 0;
     }

#if defined(CHAI_ENABLE_PICK)
     /*!
      * \brief Return the value of element i in the CopyHidingArray.
      *
      * \param index The index of the element to be fetched
      * \return The value of the i-th element in the CopyHidingArray.
      */
     CHAI_HOST_DEVICE T pick(size_type index) const
     {
#if defined(CHAI_DEVICE_COMPILE)
       return m_device_data[index];
#else
       if (m_last_touch == CPU)
       {
         return m_host_data[index];
       }
       else if (m_last_touch == GPU)
       {
         T result;
         umpire::ResourceManager::getInstance().copy(&result, m_device_data + index, sizeof(T));
         return result;
       }
       else
       {
         if (m_host_data == nullptr)
         {
           m_host_data = static_cast<T*>(umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(m_size * sizeof(T)));

         }

         return m_host_data[index];
       }
#endif
     }

     /*!
      * \brief Set the value of element i in the CopyHidingArray to be value.
      *
      * \param index The index of the element to be set
      * \param value Source location of the value
      */
     CHAI_HOST_DEVICE void set(size_type index, T value) const
     {
#if defined(CHAI_DEVICE_COMPILE)
       m_device_data[index] = value;
#else
       if (m_last_touch == CPU)
       {
         m_host_data[index] = value;
       }
       else if (m_last_touch == GPU)
       {
         umpire::ResourceManager::getInstance().copy(m_device_data + index, &value, sizeof(T));
       }
       else
       {
         if (m_host_data)
         {
           m_host_data[index] = value;
         }

         if (m_device_data)
         {
           umpire::ResourceManager::getInstance().copy(m_device_data + index, &value, sizeof(T));
         }
       }
#endif
     }

   private:
     mutable T* m_host_data = nullptr;
     mutable T* m_device_data = nullptr;
     mutable size_type m_size = 0;
     int m_host_allocator_id = getDefaultHostAllocatorId();
     int m_device_allocator_id = getDefaultDeviceAllocatorId();
     ExecutionSpace m_last_touch = NONE;
  };  // class CopyHidingArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_ARRAY_HPP
