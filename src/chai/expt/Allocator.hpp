#ifndef CHAI_ALLOCATOR_HPP
#define CHAI_ALLOCATOR_HPP

#include "chai/config.hpp"
#include "chai/expt/MemoryType.hpp"

namespace chai {
namespace expt {
   class Allocator {
      void* allocate(MemoryType type, size_t bytes);
      void deallocate(MemoryType type, void* pointer);
   };  // class Allocator
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ALLOCATOR_HPP
