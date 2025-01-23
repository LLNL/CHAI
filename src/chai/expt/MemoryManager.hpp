#ifndef CHAI_MEMORY_MANAGER_HPP
#define CHAI_MEMORY_MANAGER_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
  class MemoryManager {
      virtual void* allocate(size_t bytes) = 0;
      virtual void deallocate(void* pointer) = 0;
      virtual void get(ExecutionLocation location, bool touch) = 0;
   };  // class MemoryManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ALLOCATOR_HPP
