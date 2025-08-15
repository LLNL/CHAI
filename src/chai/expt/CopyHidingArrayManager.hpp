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
                             const umpire::Allocator& deviceAllocator);

      /*!
       * Constructs a CopyHidingArrayManager with the given Umpire allocator IDs.
       */
      CopyHidingArrayManager(int hostAllocatorID,
                             int deviceAllocatorID);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArrayManager(std::size_t size);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocators.
       */
      CopyHidingArrayManager(std::size_t size,
                             const umpire::Allocator& hostAllocator,
                             const umpire::Allocator& deviceAllocator);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocator IDs.
       */
      CopyHidingArrayManager(std::size_t size,
                             int hostAllocatorID,
                             int deviceAllocatorID);

      /*!
       * Constructs a deep copy of the given CopyHidingArrayManager.
       */
      CopyHidingArrayManager(const CopyHidingArrayManager& other);

      /*!
       * Constructs a CopyHidingArrayManager that takes ownership of the
       * resources from the given CopyHidingArrayManager.
       */
      CopyHidingArrayManager(CopyHidingArrayManager&& other) noexcept;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~CopyHidingArrayManager();

      /*!
       * \brief Copy assignment operator.
       */
      CopyHidingArrayManager& operator=(const CopyHidingArrayManager& other);

      /*!
       * \brief Move assignment operator.
       */
      CopyHidingArrayManager& operator=(CopyHidingArrayManager&& other);

      /*!
       * \brief Resize the underlying arrays.
       */
      virtual void resize(std::size_t newSize) override;

      /*!
       * \brief Get the size of the underlying arrays.
       */
      virtual std::size_t size() const override;

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       */
      virtual T* data(Context context, bool touch) override;

      /*!
       * \brief Returns the value at index i.
       *
       * Note: Use this function sparingly as it may be slow.
       *
       * \param i The index of the element to get.
       * \return The value at index i.
       */
       virtual T get(std::size_t i) const override;

       /*!
        * \brief Sets the value at index i to the specified value.
        *
        * Note: Use this function sparingly as it may be slow.
        *
        * \param i The index of the element to set.
        * \param value The value to set at index i.
        */
       virtual void set(std::size_t i, const T& value) override;

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
