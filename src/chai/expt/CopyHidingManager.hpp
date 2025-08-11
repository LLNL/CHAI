#ifndef CHAI_COPY_HIDING_MANAGER_HPP
#define CHAI_COPY_HIDING_MANAGER_HPP

#include "chai/expt/Manager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  class CopyHidingManager : public Manager {
    public:
      /*!
       * Constructs a CopyHidingManager with default allocators from Umpire
       * for the "HOST" and "DEVICE" resources.
       */
      CopyHidingManager() = default;

      /*!
       * Constructs a CopyHidingManager with the given Umpire allocators.
       */
      CopyHidingManager(const umpire::Allocator& cpuAllocator,
                        const umpire::Allocator& gpuAllocator);

      /*!
       * Constructs a CopyHidingManager with the given Umpire allocator IDs.
       */
      CopyHidingManager(int cpuAllocatorID,
                        int gpuAllocatorID);

      /*!
       * Constructs a CopyHidingManager with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      CopyHidingManager(size_type size);

      /*!
       * Constructs a CopyHidingManager with the given size using the given Umpire
       * allocators.
       */
      CopyHidingManager(size_type size,
                        const umpire::Allocator& cpuAllocator,
                        const umpire::Allocator& gpuAllocator);

      /*!
       * Constructs a CopyHidingManager with the given size using the given Umpire
       * allocator IDs.
       */
      CopyHidingManager(size_type size,
                        int cpuAllocatorID,
                        int gpuAllocatorID);

      /*!
       * Constructs a deep copy of the given CopyHidingManager.
       */
      CopyHidingManager(const CopyHidingManager& other);

      /*!
       * Constructs a CopyHidingManager that takes ownership of the
       * resources from the given CopyHidingManager.
       */
      CopyHidingManager(CopyHidingManager&& other) noexcept;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~CopyHidingManager();

      /*!
       * \brief Copy assignment operator.
       */
      CopyHidingManager& operator=(const CopyHidingManager& other);

      /*!
       * \brief Move assignment operator.
       */
      CopyHidingManager& operator=(CopyHidingManager&& other);

      /*!
       * \brief Resize the underlying arrays.
       */
      virtual void resize(size_type newSize) override;

      /*!
       * \brief Get the size of the underlying arrays.
       */
      virtual void size() const override;

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       */
      virtual void* data(bool touch) override;

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_cpu_allocator{m_resource_manager.getAllocator("HOST")};
      umpire::Allocator m_gpu_allocator{m_resource_manager.getAllocator("DEVICE")};
      size_type m_size{0};
      void* m_cpu_data{nullptr};
      void* m_gpu_data{nullptr};
      ExecutionContext m_touch{ExecutionContext::NONE};
  };  // class CopyHidingManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_MANAGER_HPP
