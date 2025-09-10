#ifndef CHAI_UNIFIED_ARRAY_MANAGER_HPP
#define CHAI_UNIFIED_ARRAY_MANAGER_HPP

#include "chai/expt/ArrayManager.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  class UnifiedArrayManager : public ArrayManager {
    public:
      UnifiedArrayManager() = default;

      explicit UnifiedArrayManager(const umpire::Allocator& allocator) :
        m_allocator{allocator}
      {
      }

      UnifiedArrayManager(std::size_t size, const umpire::Allocator& allocator) :
        m_allocator{allocator},
        m_size{size},
        m_data{m_allocator.allocate(m_size)}
      {
      }

      explicit UnifiedArrayManager(int allocatorID) :
        m_allocator{m_resource_manager.getAllocator(allocatorID)}
      {
      }

      UnifiedArrayManager(std::size_t size, int allocatorID) :
        m_allocator{m_resource_manager.getAllocator(allocatorID)},
        m_size{size},
        m_data{m_allocator.allocate(m_size)}
      {
        m_data = m_allocator.allocate(size);
      }

      UnifiedArrayManager(const UnifiedArrayManager& other) :
        m_size{other.m_size},
        m_allocator{other.m_allocator}
      {
        m_data = m_allocator.allocate(m_size * sizeof(T));
        m_execution_context_manager.setExecutionContext(ExecutionContext::DEVICE);
        m_resource_manager.copy(other.m_data, m_data, m_size * sizeof(T));
        m_execution_context_manager.setExecutionContext(ExecutionContext::NONE);
        m_modified = ExecutionContext::DEVICE;
      }

      UnifiedArrayManager(UnifiedArrayManager&& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_modified{other.m_modified},
        m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_modified = NONE;
        other.m_allocator = umpire::Allocator();
      }

      UnifiedArrayManager& operator=(const UnifiedArrayManager& other) {
        if (&other != this) { // Prevent self-assignment
          m_allocator.deallocate(m_data);

          m_size = other.m_size;
          m_allocator = other.m_allocator;
          m_data = m_allocator.allocate(m_size * sizeof(T));
          m_execution_context_manager.setExecutionContext(ExecutionContext::DEVICE);
          m_resource_manager.copy(other.m_data, m_data, m_size * sizeof(T));
          m_execution_context_manager.setExecutionContext(ExecutionContext::NONE);
          m_modified = ExecutionContext::DEVICE;
        }

        return *this;
      }

      UnifiedArrayManager& operator=(UnifiedArrayManager&& other) {
        if (&other != this) {
          m_allocator.deallocate(m_data);

          m_data = other.m_data;
          m_size = other.m_size;
          m_modified = other.m_modified;
          m_allocator = other.m_allocator;
          
          other.m_data = nullptr;
          other.m_size = 0;
          other.m_modified = ExecutionContext::NONE;
          other.m_allocator = umpire::Allocator();
        }

        return *this;
      }

      /*!
       * \brief Destructor.
       */
      ~UnifiedArrayManager() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Get the number of elements.
       */
      size_t size() const {
        return m_size;
      }

      void* data() {
        ExecutionContext executionContext = m_execution_context_manager.getExecutionContext();

        if (executionContext != m_modified) {
          m_execution_context_manager.synchronize(m_modified);
          m_modified = executionContext;
        }

        return m_data;
      }

      const T* data(ExecutionContext executionContext) const {
        if (executionContext != m_modified) {
          m_execution_context_manager.synchronize(m_modified);
          m_modified = ExecutionContext::NONE;
        }

        return m_data;
      }

      T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_modified) {
          m_execution_context_manager.synchronize(m_modified);
          m_modified = executionContext;
        }

        return m_data[i];
      }

      const T& get(ExecutionContext executionContext, size_t i) {
        if (executionContext != m_modified) {
          m_execution_context_manager.synchronize(m_modified);
          m_modified = ExecutionContext::NONE;
        }

        return m_data[i];
      }

      ExecutionContext getModified() {
        return m_modified;
      }

      umpire::Allocator getAllocator() {
        return m_allocator;
      }

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_allocator{};
      T* m_data{nullptr};
      size_t m_size{0};
      ExecutionContext m_modified{ExecutionContext::NONE};
      
      ExecutionContextManager& m_execution_context_manager{ExecutionContextManager::getInstance()};
      
  };  // class UnifiedArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_UNIFIED_ARRAY_MANAGER_HPP
