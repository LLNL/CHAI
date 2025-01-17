#ifndef CHAI_UNIFIED_MEMORY_ARRAY_HPP
#define CHAI_UNIFIED_MEMORY_ARRAY_HPP

#include "chai/config.hpp"
#include "chai/PinnedArray.hpp"

namespace chai {
namespace expt {
  template <class T>
  using UnifiedMemoryArray = PinnedArray<T>;
}  // namespace expt
}  // namespace chai

#endif  // CHAI_UNIFIED_MEMORY_ARRAY_HPP
