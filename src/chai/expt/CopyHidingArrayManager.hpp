//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_COPY_HIDING_ARRAY_MANAGER_HPP
#define CHAI_COPY_HIDING_ARRAY_MANAGER_HPP

#include "chai/expt/ArrayManager.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingArrayManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename ElementT>
  class CopyHidingArrayManager : public ArrayManager<ElementT> {
    public:
      /*!
       * Constructs a CopyHidingArrayManager with default allocators from Umpire
       * for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArrayManager() = default;

      /*!
       * Constructs a CopyHidingArrayManager with the given Umpire allocators.
       */
      CopyHidingArrayManager(const umpire::Allocator& hostAllocator,
                             const umpire::Allocator& deviceAllocator) 
        : ArrayManager<ElementT>{},
          m_host_allocator{hostAllocator},
          m_device_allocator{deviceAllocator}
      {
      }

      /*!
       * Constructs a CopyHidingArrayManager with the given Umpire allocator IDs.
       */
      CopyHidingArrayManager(int hostAllocatorID,
                             int deviceAllocatorID)
        : ArrayManager<ElementT>{},
          m_resource_manager{umpire::ResourceManager::getInstance()},
          m_host_allocator{m_resource_manager.getAllocator(hostAllocatorID)},
          m_device_allocator{m_resource_manager.getAllocator(deviceAllocatorID)}
      {
      }

      /*!
       * Constructs a CopyHidingArrayManager with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArrayManager(std::size_t size)
        : ArrayManager<ElementT>{},
          m_size{size}
      {
        // TODO: Exception handling
        m_host_data = m_host_allocator.allocate(size);
        m_device_data = m_device_allocator.allocate(size);
      }

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocators.
       */
      CopyHidingArrayManager(std::size_t size,
                             const umpire::Allocator& hostAllocator,
                             const umpire::Allocator& deviceAllocator)
        : ArrayManager<ElementT>{},
          m_host_allocator{hostAllocator},
          m_device_allocator{deviceAllocator},
          m_size{size}
      {
        // TODO: Exception handling
        m_host_data = m_host_allocator.allocate(size);
        m_device_data = m_device_allocator.allocate(size);
      }

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocator IDs.
       */
      CopyHidingArrayManager(std::size_t size,
                             int hostAllocatorID,
                             int deviceAllocatorID)
        : ArrayManager<ElementT>{},
          m_resource_manager{umpire::ResourceManager::getInstance()},
          m_host_allocator{m_resource_manager.getAllocator(hostAllocatorID)},
          m_device_allocator{m_resource_manager.getAllocator(deviceAllocatorID)},
          m_size{size}
      {
        // TODO: Exception handling
        m_host_data = m_host_allocator.allocate(size);
        m_device_data = m_device_allocator.allocate(size);
      }

      /*!
       * Constructs a deep copy of the given CopyHidingArrayManager.
       */
      CopyHidingArrayManager(const CopyHidingArrayManager& other)
        : ArrayManager<ElementT>{},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator},
          m_size{other.m_size},
          m_touch{other.m_touch}
      {
        if (other.m_host_data)
        {
          m_host_data = m_host_allocator.allocate(m_size);
          m_resource_manager.copy(m_host_data, other.m_host_data, m_size);
        }

        if (other.m_device_data)
        {
          m_device_data = m_device_allocator.allocate(m_size);
          m_resource_manager.copy(m_device_data, other.m_device_data, m_size);
        }
      }

      /*!
       * Constructs a CopyHidingArrayManager that takes ownership of the
       * resources from the given CopyHidingArrayManager.
       */
      CopyHidingArrayManager(CopyHidingArrayManager&& other) noexcept
        : ArrayManager<ElementT>{},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator},
          m_size{other.m_size},
          m_touch{other.m_touch},
          m_host_data{other.m_host_data},
          m_device_data{other.m_device_data}
      {
        other.m_size = 0;
        other.m_host_data = nullptr;
        other.m_device_data = nullptr;
        other.m_touch = ExecutionContext::NONE;
      }

      /*!
       * \brief Virtual destructor.
       */
      virtual ~CopyHidingArrayManager()
      {
        if (m_host_data) {
          m_host_allocator.deallocate(m_host_data);
        }
        if (m_device_data) {
          m_device_allocator.deallocate(m_device_data);
        }
      }

      /*!
       * \brief Copy assignment operator.
       */
      CopyHidingArrayManager& operator=(const CopyHidingArrayManager& other)
      {
        if (this != &other)
        {
          // Copy-assign or copy members
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;
          m_touch = other.m_touch;

          // Allocate new resources before releasing old ones for strong exception safety
          void* new_host_data = nullptr;
          void* new_device_data = nullptr;

          if (other.m_host_data)
          {
            new_host_data = m_host_allocator.allocate(other.m_size);
            m_resource_manager.copy(new_host_data, other.m_host_data, other.m_size);
          }

          if (other.m_device_data)
          {
            new_device_data = m_device_allocator.allocate(other.m_size);
            m_resource_manager.copy(new_device_data, other.m_device_data, other.m_size);
          }

          // Clean up old resources
          if (m_host_data)
          {
            m_host_allocator.deallocate(m_host_data);
          }

          if (m_device_data)
          {
            m_device_allocator.deallocate(m_device_data);
          }

          // Assign new resources and size
          m_host_data = new_host_data;
          m_device_data = new_device_data;
          m_size = other.m_size;
        }

        return *this;
      }

      /*!
       * \brief Move assignment operator.
       */
      CopyHidingArrayManager& operator=(CopyHidingArrayManager&& other) noexcept
      {
        if (this != &other)
        {
          // Release any resources currently held
          if (m_host_data)
          {
            m_host_allocator.deallocate(m_host_data);
            m_host_data = nullptr;
          }
          if (m_device_data)
          {
            m_device_allocator.deallocate(m_device_data);
            m_device_data = nullptr;
          }

          // Move-assign or copy members
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;
          m_size = other.m_size;
          m_host_data = other.m_host_data;
          m_device_data = other.m_device_data;
          m_touch = other.m_touch;

          // Null out other's pointers and reset size
          other.m_host_data = nullptr;
          other.m_device_data = nullptr;
          other.m_size = 0;
          other.m_touch = ExecutionContext::NONE;
        }
        return *this;
      }

      /*!
       * \brief Resize the underlying arrays.
       */
      virtual void resize(std::size_t newSize) override
      {
        if (newSize != m_size)
        {
          if (m_touch == ExecutionContext::CPU)
          {
            m_resource_manager.reallocate(m_host_data, newSize);

            if (m_device_data)
            {
              m_resource_manager.deallocate(m_device_data);
              m_device_data = m_device_allocator.allocate(newSize);
            }
          }
          else if (m_touch == ExecutionContext::GPU)
          {
            m_resource_manager.reallocate(m_device_data, newSize);

            if (m_host_data)
            {
              m_resource_manager.deallocate(m_host_data);
              m_host_data = m_host_allocator.allocate(newSize);
            }
          }
          else
          {
            if (m_device_data)
            {
              m_resource_manager.reallocate(m_device_data, newSize);
            }

            if (m_host_data)
            {
              m_resource_manager.reallocate(m_host_data, newSize);
            }
          }
          m_size = newSize;
        }
      }

      /*!
       * \brief Get the size of the underlying arrays.
       */
      virtual std::size_t size() const override
      {
        return m_size;
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       */
      virtual ElementT* data(Context context, bool touch) override
      {
        if (context == ExecutionContext::CPU)
        {
          if (!m_host_data)
          {
            m_host_data = m_host_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::GPU)
          {
            m_resource_manager.copy(m_host_data, m_device_data, m_size);
            m_touch = ExecutionContext::NONE;
          }

          if (touch)
          {
            m_touch = ExecutionContext::CPU;
          }

          return static_cast<ElementT*>(m_host_data);
        }
        else if (context == ExecutionContext::GPU)
        {
          if (!m_device_data)
          {
            m_device_data = m_device_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::CPU)
          {
            m_resource_manager.copy(m_device_data, m_host_data, m_size);
            m_touch = ExecutionContext::NONE;
          }

          if (touch)
          {
            m_touch = ExecutionContext::GPU;
          }

          return static_cast<ElementT*>(m_device_data);
        }
        else
        {
          return nullptr;
        }
      }

      /*!
       * \brief Returns the value at index i.
       *
       * Note: Use this function sparingly as it may be slow.
       *
       * \param i The index of the element to get.
       * \return The value at index i.
       */
       virtual ElementT get(std::size_t i) const override
       {
         // Implementation needed
         // For now, just returning a default value
         return ElementT{};
       }

       /*!
        * \brief Sets the value at index i to the specified value.
        *
        * Note: Use this function sparingly as it may be slow.
        *
        * \param i The index of the element to set.
        * \param value The value to set at index i.
        */
       virtual void set(std::size_t i, const ElementT& value) override
       {
         // Implementation needed
       }

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_host_allocator{m_resource_manager.getAllocator("HOST")};
      umpire::Allocator m_device_allocator{m_resource_manager.getAllocator("DEVICE")};
      std::size_t m_size{0};
      void* m_host_data{nullptr};
      void* m_device_data{nullptr};
      ExecutionContext m_touch{ExecutionContext::NONE};
  };  // class CopyHidingArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_ARRAY_MANAGER_HPP
