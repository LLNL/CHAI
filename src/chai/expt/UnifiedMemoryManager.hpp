#ifndef CHAI_UNIFIED_MEMORY_MANAGER_HPP
#define CHAI_UNIFIED_MEMORY_MANAGER_HPP

#include "chai/expt/ExecutionContext.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  class UnifiedMemoryManager
    public:
      UnifiedMemoryManager() = default;

      explicit UnifiedMemoryManager(const umpire::Allocator& allocator) :
        m_allocator{allocator}
      {
      }

      UnifiedMemoryManager(std::size_t size, const umpire::Allocator& allocator) :
        m_allocator{allocator},
        m_size{size},
        m_data{m_allocator.allocate(m_size)}
      {
      }

      explicit UnifiedMemoryManager(int allocatorID) :
        m_allocator{m_resource_manager.getAllocator(allocatorID)}
      {
      }

      UnifiedMemoryManager(std::size_t size, int allocatorID) :
        m_allocator{m_resource_manager.getAllocator(allocatorID)},
        m_size{size},
        m_data{m_allocator.allocate(m_size)}
      {
        m_data = m_allocator.allocate(size);
      }

      UnifiedMemoryManager(const UnifiedMemoryManager& other) :
        m_size{other.m_size},
        m_allocator{other.m_allocator}
      {
        m_data = m_allocator.allocate(m_size * sizeof(T));
        m_execution_context_manager.setExecutionContext(ExecutionContext::DEVICE);
        m_resource_manager.copy(other.m_data, m_data, m_size * sizeof(T));
        m_execution_context_manager.setExecutionContext(ExecutionContext::NONE);
        m_last_execution_context = ExecutionContext::DEVICE;
      }

      UnifiedMemoryManager(UnifiedMemoryManager&& other) :
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

      UnifiedMemoryManager& operator=(const UnifiedMemoryManager& other) {
        if (&other != this) { // Prevent self-assignment
          m_allocator.deallocate(m_data);

          m_size = other.m_size;
          m_allocator = other.m_allocator;
          m_data = m_allocator.allocate(m_size * sizeof(T));
          m_execution_context_manager.setExecutionContext(ExecutionContext::DEVICE);
          m_resource_manager.copy(other.m_data, m_data, m_size * sizeof(T));
          m_execution_context_manager.setExecutionContext(ExecutionContext::NONE);
          m_last_execution_context = ExecutionContext::DEVICE;
        }

        return *this;
      }

      UnifiedMemoryManager& operator=(UnifiedMemoryManager&& other) {
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
      ~UnifiedMemoryManager() {
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
          m_execution_context_manager.synchronize(m_last_execution_context);
          m_last_execution_context = executionContext;
        }

        return m_data;
      }

      const T* data(ExecutionContext executionContext) const {
        if (executionContext != m_last_execution_context) {
          m_execution_context_manager.synchronize(m_last_execution_context);
          m_last_execution_context = ExecutionContext::NONE;
        }

        return m_data;
      }

      T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_last_execution_context) {
          m_execution_context_manager.synchronize(m_last_execution_context);
          m_last_execution_context = executionContext;
        }

        return m_data[i];
      }

      const T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_last_execution_context) {
          m_execution_context_manager.synchronize(m_last_execution_context);
          m_last_execution_context = ExecutionContext::NONE;
        }

        return m_data[i];
      }

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_allocator{};
      T* m_data{nullptr};
      size_t m_size{0};
      ExecutionContext m_last_execution_context{ExecutionContext::NONE};
      
      ExecutionContextManager& m_execution_context_manager{ExecutionContextManager::getInstance()};
      
  };  // class UnifiedMemoryManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_UNIFIED_MEMORY_MANAGER_HPP
