#ifndef CHAI_UNIFIED_ARRAY_HPP
#define CHAI_UNIFIED_ARRAY_HPP

#include "chai/expt/ExecutionContext.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class UnifiedArray
   *
   * \brief A container for managing the lifetime and coherence of a
   *        unified memory array, meaning an array with a single address
   *        that is accessible from all processors/devices in a system.
   *
   * This container should be used in tandem with the ExecutionContextManager.
   * Together, they provide a programming model where work (e.g. a kernel)
   * is generally performed asynchronously, with synchronization occurring
   * only as needed for coherence of the array. For example, if the array is
   * written to in an asynchronize kernel on a GPU, then the GPU will be
   * synchronized if the array needs to be accessed on the CPU.
   *
   * This model works well for APUs where the CPU and GPU have the same
   * physical memory. It also works for pinned (i.e. page-locked) memory
   * and in some cases for pageable memory, though no pre-fetching is
   * performed.
   *
   * Example:
   *
   * \code
   * // Create a UnifiedArray with size 100 and default allocator
   * int size = 10000;
   * UnifiedArray<int> array(size);
   *
   * // Access elements on the device
   * std::span<int> device_view(array.data(ExecutionContext::DEVICE, array.size());
   *
   * // Launch a kernel that modifies device_view.
   * // Note that this example relies on c++20 and the ability to use constexpr
   * // host code on the device.
   *
   * // Access elements on the host. This will synchronize the device.
   * std::span<int> host_view(array.data(ExecutionContext::HOST), array.size());
   *
   * for (int i = 0; i < size; ++i) {
   *   std::cout << host_view[i] << "\n";
   * }
   *
   * // Access and modify individual elements in the container.
   * // This should be used sparingly or it will tank performance.
   * // Getting the last element after performing a scan is one use case.
   * array.get(ExecutionContext::HOST, size - 1) = 10;
   * \endcode
   */
  template <typename T>
  class UnifiedArray
    public:
      UnifiedArray() = default;

      explicit UnifiedArray(const umpire::Allocator& allocator) :
        m_allocator{allocator}
      {
      }

      UnifiedArray(std::size_t size, const umpire::Allocator& allocator) :
        m_allocator{allocator}
      {
        resize(size);
      }

      UnifiedArray(const UnifiedArray& other)
        : m_allocator{other.m_allocator}
      {
        resize(other.m_size);
        ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::DEVICE);
        umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size * sizeof(T));
        ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::NONE);
        m_modified = ExecutionContext::DEVICE;
      }

      UnifiedArray(UnifiedArray&& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_modified{other.m_modified},
        m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_modified = ExecutionContext::NONE;
      }

      ~UnifiedArray()
      {
        m_allocator.deallocate(m_data);
      }

      UnifiedArray& operator=(const UnifiedArray& other) {
        if (&other != this) { // Prevent self-assignment
          m_allocator.deallocate(m_data);

          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = static_cast<T*>(m_allocator.allocate(m_size * sizeof(T)));

          ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::DEVICE);
          umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size * sizeof(T));
          ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::NONE);
          
          m_modified = ExecutionContext::DEVICE;
        }

        return *this;
      }

      UnifiedArray& operator=(UnifiedArray&& other) {
        if (&other != this) {
          m_allocator.deallocate(m_data);

          m_data = other.m_data;
          m_size = other.m_size;
          m_modified = other.m_modified;
          m_allocator = other.m_allocator;
          
          other.m_data = nullptr;
          other.m_size = 0;
          other.m_modified = ExecutionContext::NONE;
        }

        return *this;
      }

      void resize(std::size_t newSize)
      {
        if (newSize != m_size)
        {
          T* newData = nullptr;

          if (newSize > 0)
          {
            std::size_t newSizeBytes = newSize * sizeof(T);
            ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::DEVICE);
            newData = static_cast<T*>(m_allocator.allocate(newSizeBytes));
            umpire::ResourceManager::getInstance().copy(other.m_data, m_data, std::min(newSizeBytes, m_size * sizeof(T)));
            ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::NONE);
            m_modified = ExecutionContext::DEVICE;
          }

          m_allocator.deallocate(m_data);
          m_data = newData;
          m_size = newSize;
        }
      }

      void free()
      {
        m_allocator.deallocate(m_data);
        m_data = nullptr;
        m_size = 0;
        m_modified = ExecutionContext::NONE;
      }

      /*!
       * \brief Get the number of elements.
       */
      size_t size() const
      {
        return m_size;
      }

      T* data()
      {
        ExecutionContext executionContext =
          ExecutionContextManager::getInstance().getExecutionContext();

        if (executionContext != m_modified) {
          ExecutionContextManager::getInstance().synchronize(m_modified);
          m_modified = executionContext;
        }

        return m_data;
      }

      const T* data() const {
        ExecutionContext executionContext =
          ExecutionContextManager::getInstance().getExecutionContext();

        if (executionContext != m_modified) {
          ExecutionContextManager::getInstance().synchronize(m_modified);
          m_modified = ExecutionContext::NONE;
        }

        return m_data;
      }

      T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_modified) {
          ExecutionContextManager::getInstance().synchronize(m_modified);
          m_modified = executionContext;
        }

        return m_data[i];
      }

      const T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_modified) {
          ExecutionContextManager::getInstance().synchronize(m_modified);
          m_modified = ExecutionContext::NONE;
        }

        return m_data[i];
      }

    private:
      T* m_data{nullptr};
      size_t m_size{0};
      ExecutionContext m_modified{ExecutionContext::NONE};
      umpire::Allocator m_allocator{};
  };  // class UnifiedArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_UNIFIED_ARRAY_HPP
