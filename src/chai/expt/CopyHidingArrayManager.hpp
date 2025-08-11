#ifndef CHAI_COPY_HIDING_ARRAY_MANAGER_HPP
#define CHAI_COPY_HIDING_ARRAY_MANAGER_HPP

#include "chai/expt/Manager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingArrayManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  class CopyHidingArrayManager : public Manager {
    public:
      /*!
       * Constructs a CopyHidingArrayManager with default allocators from Umpire
       * for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArrayManager() = default;

      /*!
       * Constructs a CopyHidingArrayManager with the given Umpire allocators.
       */
      CopyHidingArrayManager(const umpire::Allocator& cpuAllocator,
                        const umpire::Allocator& gpuAllocator);

      /*!
       * Constructs a CopyHidingArrayManager with the given Umpire allocator IDs.
       */
      CopyHidingArrayManager(int cpuAllocatorID,
                        int gpuAllocatorID);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using default allocators
       * from Umpire for the "HOST" and "DEVICE" resources.
       */
      CopyHidingArrayManager(size_type size);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocators.
       */
      CopyHidingArrayManager(size_type size,
                        const umpire::Allocator& cpuAllocator,
                        const umpire::Allocator& gpuAllocator);

      /*!
       * Constructs a CopyHidingArrayManager with the given size using the given Umpire
       * allocator IDs.
       */
      CopyHidingArrayManager(size_type size,
                        int cpuAllocatorID,
                        int gpuAllocatorID);

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
  };  // class CopyHidingArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_ARRAY_MANAGER_HPP
