//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_PINNED_ARRAY_MANAGER_HPP
#define CHAI_PINNED_ARRAY_MANAGER_HPP

#include "chai/expt/ArrayManager.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class PinnedArrayManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename ElementT>
  class PinnedArrayManager : public ArrayManager<ElementT> {
    public:
      /*!
       * Constructs a PinnedArrayManager with default allocators from Umpire
       * for the "HOST" and "DEVICE" resources.
       */
      PinnedArrayManager() = default;

      /*!
       * Constructs a PinnedArrayManager with the given Umpire allocators.
       */
      PinnedArrayManager(const umpire::Allocator& allocator) 
        : ArrayManager<ElementT>{},
          m_allocator{allocator}
      {
      }

      /*!
       * Constructs a PinnedArrayManager with the given Umpire allocator IDs.
       */
      PinnedArrayManager(int allocatorID)
        : ArrayManager<ElementT>{},
          m_resource_manager{umpire::ResourceManager::getInstance()},
          m_allocator{m_resource_manager.getAllocator(allocatorID)}
      {
      }

      /*!
       * Constructs a PinnedArrayManager with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      PinnedArrayManager(std::size_t size)
        : ArrayManager<ElementT>{},
          m_size{size}
      {
        // TODO: Exception handling
        m_data = static_cast<ElementT*>(m_allocator.allocate(size*sizeof(ElementT));
      }

      /*!
       * Constructs a PinnedArrayManager with the given size using the given Umpire
       * allocators.
       */
      PinnedArrayManager(std::size_t size,
                         const umpire::Allocator& allocator)
        : ArrayManager<ElementT>{},
          m_allocator{allocator},
          m_size{size}
      {
        // TODO: Exception handling
        m_data = static_cast<ElementT*>(m_allocator.allocate(size*sizeof(ElementT));
      }

      /*!
       * Constructs a PinnedArrayManager with the given size using the given Umpire
       * allocator IDs.
       */
      PinnedArrayManager(std::size_t size,
                         int allocatorID)
        : ArrayManager<ElementT>{},
          m_resource_manager{umpire::ResourceManager::getInstance()},
          m_allocator{m_resource_manager.getAllocator(allocatorID)},
          m_size{size}
      {
        // TODO: Exception handling
        static_cast<ElementT*>(m_allocator.allocate(size*sizeof(ElementT));
      }

      /*!
       * Constructs a deep copy of the given PinnedArrayManager.
       */
      PinnedArrayManager(const PinnedArrayManager& other)
        : ArrayManager<ElementT>{},
          m_allocator{other.m_allocator},
          m_size{other.m_size},
          m_touch{other.m_touch}
      {
        if (other.m_data)
        {
          m_data = m_allocator.allocate(m_size);
          m_resource_manager.copy(m_data, other.m_data, m_size*sizeof(ElementT));
          // TODO: The copy could potentially change in which space the last touch occurs
        }
      }

      /*!
       * Constructs a PinnedArrayManager that takes ownership of the
       * resources from the given PinnedArrayManager.
       */
      PinnedArrayManager(PinnedArrayManager&& other) noexcept
        : ArrayManager<ElementT>{},
          m_allocator{other.m_allocator},
          m_size{other.m_size},
          m_touch{other.m_touch},
          m_data{other.m_data}
      {
        other.m_size = 0;
        other.m_data = nullptr;
        other.m_touch = NONE;
      }

      /*!
       * \brief Virtual destructor.
       */
      virtual ~PinnedArrayManager()
      {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Copy assignment operator.
       */
      PinnedArrayManager& operator=(const PinnedArrayManager& other)
      {
        if (this != &other)
        {
          // Copy-assign or copy members
          m_allocator = other.m_allocator;
          m_touch = other.m_touch;

          // Allocate new resources before releasing old ones for strong exception safety
          void* new_data = nullptr;

          if (other.m_data)
          {
            new_data = static_cast<ElementT*>(m_allocator.allocate(other.m_size*sizeof(ElementT)));
            m_resource_manager.copy(new_data, other.m_data, other.m_size*sizeof(ElementT));
            // TODO: The copy operation could change m_touch
          }

          // Clean up old resources
          if (m_data)
          {
            m_allocator.deallocate(m_data);
          }

          // Assign new resources and size
          m_data = new_data;
          m_size = other.m_size;
        }

        return *this;
      }

      /*!
       * \brief Move assignment operator.
       */
      PinnedArrayManager& operator=(PinnedArrayManager&& other) noexcept
      {
        if (this != &other)
        {
          // Release any resources currently held
          if (m_data)
          {
            m_allocator.deallocate(m_data);
          }

          // Move-assign or copy members
          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = other.m_data;
          m_touch = other.m_touch;

          // Null out other's pointers and reset size
          other.m_data = nullptr;
          other.m_size = 0;
          other.m_touch = NONE;
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
          // TODO: Is any synchronization needed?
          m_resource_manager.reallocate(m_data, newSize);
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
        ElementT* result{nullptr};

        if (context == HOST)
        {
          if (m_touch == DEVICE)
          {
            m_context_manager.synchronize(DEVICE);
            m_touch = NONE;
          }

          if (touch)
          {
            m_touch = HOST;
          }

          result = m_data;
        }
        else if (context == DEVICE)
        {
          if (m_touch == HOST)
          {
            // TODO: Should we call m_context_manager.synchronize(HOST)? Would support host openmp.
            m_touch = NONE;
          }

          if (touch)
          {
            m_touch = DEVICE;
          }

          result = m_data;
        }

        return result;
      }

      /*!
       * \brief Returns the value at index i.
       *
       * Note: Use this function sparingly as it may be slow.
       *
       * \param i The index of the element to get.
       * \return The value at index i.
       */
      virtual ElementT get(std::size_t i) const override {
        m_context_manager.synchronize(m_touch);
        m_touch = NONE;
        return m_data[i];
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
        m_context_manager.synchronize(m_touch);
        m_touch = HOST;
        m_data[i] = value;
      }

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_allocator{m_resource_manager.getAllocator("DEVICE")};
      std::size_t m_size{0};
      ElementT* m_data{nullptr};
      ExecutionContext m_touch{NONE};
  };  // class PinnedArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_MANAGER_HPP
