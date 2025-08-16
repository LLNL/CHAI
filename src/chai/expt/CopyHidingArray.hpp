#ifndef CHAI_COPY_HIDING_ARRAY_HPP
#define CHAI_COPY_HIDING_ARRAY_HPP

#include "umpire/ResourceManager.hpp"

// TODO: Determine how to specify starting execution space

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingArray
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename ElementType>
  class CopyHidingArray
    public:
      /*!
       * Constructs a CopyHidingArray with default allocators from Umpire
       * for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArray() = default;

      /*!
       * Constructs a CopyHidingArray with the given Umpire allocators.
       */
      CopyHidingArray(const umpire::Allocator& cpuAllocator,
                      const umpire::Allocator& gpuAllocator) :
        m_cpu_allocator{cpuAllocator},
        m_gpu_allocator{gpuAllocator}
      { 
      }

      /*!
       * Constructs a CopyHidingArray with the given Umpire allocator IDs.
       */
      CopyHidingArray(int cpuAllocatorID,
                      int gpuAllocatorID) :
        m_resource_manager{umpire::ResourceManager::getInstance()},
        m_cpu_allocator{m_resource_manager.getAllocator(cpuAllocatorID)},
        m_gpu_allocator{m_resource_manager.getAllocator(gpuAllocatorID)}
      {
      }

      /*!
       * Constructs a CopyHidingArray with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArray(size_type size) :
        m_size{size}
      {
        // TODO: Exception handling
        m_cpu_data = m_cpu_allocator.allocate(size);
        m_gpu_data = m_gpu_allocator.allocate(size);
      }

      /*!
       * Constructs a CopyHidingArray with the given size using the given Umpire
       * allocators.
       */
      CopyHidingArray(size_type size,
                      const umpire::Allocator& cpuAllocator,
                      const umpire::Allocator& gpuAllocator) :
        m_cpu_allocator{cpuAllocator},
        m_gpu_allocator{gpuAllocator},
        m_size{size}
      {
        // TODO: Exception handling
        m_cpu_data = m_cpu_allocator.allocate(size);
        m_gpu_data = m_gpu_allocator.allocate(size);
      }

      /*!
       * Constructs a CopyHidingArray with the given size using the given Umpire
       * allocator IDs.
       */
      CopyHidingArray(size_type size,
                      int cpuAllocatorID,
                      int gpuAllocatorID) :
        m_resource_manager{umpire::ResourceManager::getInstance()},
        m_cpu_allocator{m_resource_manager.getAllocator(cpuAllocatorID)},
        m_gpu_allocator{m_resource_manager.getAllocator(gpuAllocatorID)},
        m_size{size}
      {
        // TODO: Exception handling
        m_cpu_data = m_cpu_allocator.allocate(size);
        m_gpu_data = m_gpu_allocator.allocate(size);
      }

      /*!
       * Constructs a deep copy of the given CopyHidingArray.
       */
      CopyHidingArray(const CopyHidingArray& other) :
        m_cpu_allocator{other.m_cpu_allocator},
        m_gpu_allocator{other.m_gpu_allocator},
        m_size{other.m_size},
        m_touch{other.m_touch}
      {
        if (other.m_cpu_data)
        {
          m_cpu_data = m_cpu_allocator.allocate(m_size);
          m_resourceManager.copy(m_cpu_data, other.m_cpu_data, m_size);
        }

        if (other.m_gpu_data)
        {
          m_gpu_data = m_gpu_allocator.allocate(m_size);
          m_resourceManager.copy(m_gpu_data, other.m_gpu_data, m_size);
        }
      }

      /*!
       * Constructs a CopyHidingArray that takes ownership of the
       * resources from the given CopyHidingArray.
       */
      CopyHidingArray(CopyHidingArray&& other) :
        m_cpu_allocator{other.m_cpu_allocator},
        m_gpu_allocator{other.m_gpu_allocator},
        m_size{other.m_size},
        m_touch{other.m_touch},
        m_cpu_data{other.m_cpu_data},
        m_gpu_data{other.m_gpu_data}
      {
        other.m_size = 0;
        other.m_cpu_data = nullptr;
        other.m_gpu_data = nullptr;
        other.m_touch = ExecutionContext::NONE;
      }

      /*!
       * \brief Virtual destructor.
       */
      ~CopyHidingArray()
      {
        m_cpu_allocator.deallocate(m_cpu_data);
        m_gpu_allocator.deallocate(m_gpu_data);
      }

      /*!
       * \brief Copy assignment operator.
       */
      CopyHidingArray& operator=(const CopyHidingArray& other)
      {
        if (this != &other)
        {
          // Copy-assign base class if needed (uncomment if Manager is copy-assignable)
          // Manager::operator=(other);

          // Copy-assign or copy members
          m_cpu_allocator = other.m_cpu_allocator;
          m_gpu_allocator = other.m_gpu_allocator;
          m_touch = other.m_touch;

          // Allocate new resources before releasing old ones for strong exception safety
          void* new_cpu_data = nullptr;
          void* new_gpu_data = nullptr;

          if (other.m_cpu_data)
          {
            new_cpu_data = m_cpu_allocator.allocate(other.m_size);
            m_resourceManager.copy(new_cpu_data, other.m_cpu_data, other.m_size);
          }

          if (other.m_gpu_data)
          {
            new_gpu_data = m_gpu_allocator.allocate(other.m_size);
            m_resourceManager.copy(new_gpu_data, other.m_gpu_data, other.m_size);
          }

          // Clean up old resources
          if (m_cpu_data)
          {
            m_cpu_allocator.deallocate(m_cpu_data, m_size);
          }

          if (m_gpu_data)
          {
            m_gpu_allocator.deallocate(m_gpu_data, m_size);
          }

          // Assign new resources and size
          m_cpu_data = new_cpu_data;
          m_gpu_data = new_gpu_data;
          m_size = other.m_size;
        }

        return *this;
      }

      /*!
       * \brief Move assignment operator.
       */
      CopyHidingArray& operator=(CopyHidingArray&& other)
      {
        if (this != &other)
        {
          // Release any resources currently held
          if (m_cpu_data)
          {
            m_cpu_allocator.deallocate(m_cpu_data, m_size);
            m_cpu_data = nullptr;
          }

          if (m_gpu_data)
          {
            m_gpu_allocator.deallocate(m_gpu_data, m_size);
            m_gpu_data = nullptr;
          }

          // Move-assign base class if needed (uncomment if Manager is move-assignable)
          // Manager::operator=(std::move(other));

          // Move-assign or copy members
          m_cpu_allocator = other.m_cpu_allocator;
          m_gpu_allocator = other.m_gpu_allocator;
          m_size = other.m_size;
          m_cpu_data = other.m_cpu_data;
          m_gpu_data = other.m_gpu_data;
          m_touch = other.m_touch;

          // Null out other's pointers and reset size
          other.m_cpu_data = nullptr;
          other.m_gpu_data = nullptr;
          other.m_size = 0;
          other.m_touch = ExecutionContext::NONE;
        }

        return *this;
      }

      /*!
       * \brief Resize the underlying arrays.
       */
      void resize(size_type newSize)
      {
        if (newSize != m_size)
        {
          if (m_touch == ExecutionContext::CPU)
          {
            m_resource_manager.reallocate(m_cpu_pointer, newSize);

            if (m_gpu_pointer)
            {
              m_resource_manager.deallocate(m_gpu_pointer);
              m_gpu_pointer = m_gpu_allocator.allocate(newSize);
            }
          }
          else if (m_touch == ExecutionContext::GPU)
          {
            m_resource_manager.reallocate(m_gpu_pointer, newSize);

            if (m_cpu_pointer)
            {
              m_resource_manager.deallocate(m_cpu_pointer);
              m_cpu_pointer = m_cpu_allocator.allocate(newSize);
            }
          }
          else
          {
            if (m_gpu_pointer)
            {
              m_resource_manager.reallocate(m_gpu_pointer, newSize);
            }

            if (m_cpu_pointer)
            {
              m_resource_manager.reallocate(m_cpu_pointer, newSize);
            }
          }
        }
      }

      /*!
       * \brief Get the size of the underlying arrays.
       */
      size_type size() const
      {
        return m_size;
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       */
      T* data(ExecutionContext context)
      {
        if (context == ExecutionContext::CPU)
        {
          if (!m_cpu_data)
          {
            m_cpu_data = m_cpu_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::GPU)
          {
            m_resource_manager.copy(m_cpu_data, m_gpu_data, m_size);
            m_touch = ExecutionContext::NONE;
          }

          if (touch)
          {
            m_touch = ExecutionContext::CPU;
          }

          return m_cpu_data;
        }
        else if (context == ExecutionContext::GPU)
        {
          if (!m_gpu_data)
          {
            m_gpu_data = m_gpu_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::CPU)
          {
            m_resource_manager.copy(m_gpu_data, m_cpu_data, m_size);
            m_touch = ExecutionContext::NONE;
          }

          if (touch)
          {
            m_touch = ExecutionContext::GPU;
          }

          return m_gpu_data;
        }
        else
        {
          return nullptr;
        }
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       */
       const T* data(ExecutionContext context) const
       {
         if (context == ExecutionContext::CPU)
         {
           if (!m_cpu_data)
           {
             m_cpu_data = m_cpu_allocator.allocate(m_size);
           }
 
           if (m_touch == ExecutionContext::GPU)
           {
             m_resource_manager.copy(m_cpu_data, m_gpu_data, m_size);
             m_touch = ExecutionContext::NONE;
           }
 
           return m_cpu_data;
         }
         else if (context == ExecutionContext::GPU)
         {
           if (!m_gpu_data)
           {
             m_gpu_data = m_gpu_allocator.allocate(m_size);
           }
 
           if (m_touch == ExecutionContext::CPU)
           {
             m_resource_manager.copy(m_gpu_data, m_cpu_data, m_size);
             m_touch = ExecutionContext::NONE;
           }
 
           return m_gpu_data;
         }
         else
         {
           return nullptr;
         }
       }

#if 0
      /*!
       * \brief Get the i-th element.
       *
       * \warning Use sparingly, as coherence must be checked.
       */
       ElementType getElement(size_type i) const
       {
          if (m_touch == ExecutionContext::GPU)
          {
            // Copy m_gpu_data[i] to host
          }
          else
          {
            return m_cpu_data[i];
          }
          
          return m_cpu_data[i];
       }

       void setElement(size_type i, const ElementType& value)
       {
         if (m_touch == ExecutionContext::GPU)
         {
           // Copy value to m_gpu_data[i]
         }
         else
         {
           m_cpu_data[i] = value;
         }
       }
#endif

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_cpu_allocator{m_resource_manager.getAllocator("HOST")};
      umpire::Allocator m_gpu_allocator{m_resource_manager.getAllocator("DEVICE")};
      size_type m_size{0};
      ElementType* m_cpu_data{nullptr};
      ElementType* m_gpu_data{nullptr};
      ExecutionContext m_touch{ExecutionContext::NONE};
  };  // class CopyHidingArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_ARRAY_HPP
