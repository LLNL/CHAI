#ifndef CHAI_PINNED_ARRAY_CONTAINER_HPP
#define CHAI_PINNED_ARRAY_CONTAINER_HPP

#include "chai/expt/ExecutionContext.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class PinnedArrayContainer
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename T>
  class PinnedArrayContainer
    public:
      PinnedArrayContainer() = default;

      explicit PinnedArrayContainer(const umpire::Allocator& allocator) :
        m_allocator{allocator}
      {
      }

      PinnedArrayContainer(std::size_t size, const umpire::Allocator& allocator) :
        m_size{size},
        m_allocator{allocator}
      {
        m_data = m_allocator.allocate(m_size * sizeof(T));
        // TODO: Initialization
      }

      explicit PinnedArrayContainer(int allocatorID) :
        m_allocator{umpire::ResourceManager::getInstance().getAllocator(allocatorID)}
      {
      }

      PinnedArrayContainer(std::size_t size, int allocatorID) :
        m_size{size},
        m_allocator{umpire::ResourceManager::getInstance().getAllocator(allocatorID)}
      {
        m_data = m_allocator.allocate(m_size * sizeof(T));
        // TODO: Initialization
      }

      PinnedArrayContainer(const PinnedArrayContainer& other) :
        m_size{other.m_size},
        m_allocator{other.m_allocator}
      {
        m_data = m_allocator.allocate(m_size * sizeof(T));
        ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::DEVICE);
        umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size * sizeof(T));
        ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::NONE);
        m_last_execution_context = ExecutionContext::DEVICE;
      }

      PinnedArrayContainer(PinnedArrayContainer&& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_last_execution_context{other.m_last_execution_context},
        m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_last_execution_context = NONE;
        other.m_allocator = umpire::Allocator();
      }

      PinnedArrayContainer& operator=(const PinnedArrayContainer& other) {
        if (&other != this) { // Prevent self-assignment
          m_allocator.deallocate(m_data);

          m_size = other.m_size;
          m_allocator = other.m_allocator;
          m_data = m_allocator.allocate(m_size * sizeof(T));
          ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::DEVICE);
          umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size * sizeof(T));
          ExecutionContextManager::getInstance().setExecutionContext(ExecutionContext::NONE);
          m_last_execution_context = ExecutionContext::DEVICE;
        }

        return *this;
      }

      PinnedArrayContainer& operator=(PinnedArrayContainer&& other) {
        if (&other != this) {
          m_allocator.deallocate(m_data);

          m_data = other.m_data;
          m_size = other.m_size;
          m_last_execution_context = other.m_last_execution_context;
          m_allocator = other.m_allocator;
          
          other.m_data = nullptr;
          other.m_size = 0;
          other.m_last_execution_context = ExecutionContext::NONE;
          other.m_allocator = umpire::Allocator();
        }

        return *this;
      }

      /*!
       * \brief Destructor.
       */
      ~PinnedArrayContainer() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Get the number of elements.
       */
      size_t size() const {
        return m_size;
      }

      T* data(ExecutionContext executionContext) {
        if (executionContext != m_last_execution_context) {
          ExecutionContextManager::getInstance().synchronize(m_last_execution_context);
          m_last_execution_context = executionContext;
        }

        return m_data;
      }

      const T* data(ExecutionContext executionContext) const {
        if (executionContext != m_last_execution_context) {
          ExecutionContextManager::getInstance().synchronize(m_last_execution_context);
          m_last_execution_context = ExecutionContext::NONE;
        }

        return m_data;
      }

      T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_last_execution_context) {
          ExecutionContextManager::getInstance().synchronize(m_last_execution_context);
          m_last_execution_context = executionContext;
        }

        return m_data[i];
      }

      const T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_last_execution_context) {
          ExecutionContextManager::getInstance().synchronize(m_last_execution_context);
          m_last_execution_context = ExecutionContext::NONE;
        }

        return m_data[i];
      }

    private:
      T* m_data{nullptr};
      size_t m_size{0};
      ExecutionContext m_last_execution_context{ExecutionContext::NONE};
      umpire::Allocator m_allocator{};
  };  // class PinnedArrayContainer
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_CONTAINER_HPP
