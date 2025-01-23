#ifndef CHAI_HOST_MEMORY_MANAGER_HPP
#define CHAI_HOST_MEMORY_MANAGER_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
  class HostMemoryManager {
    public:
      virtual void* allocate(size_t bytes) 
      virtual void deallocate(void* pointer) = 0;
      virtual void get(ExecutionLocation location, bool touch) = 0;

    private:
      void* m_data = nullptr;
      size_t m_bytes = 0;
      umpire::Allocator* m_allocator = nullptr;
  };  // class HostMemoryManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ALLOCATOR_HPP
