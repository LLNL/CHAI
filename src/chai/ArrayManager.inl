//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"

#include <iostream>

#include "umpire/ResourceManager.hpp"

#if defined(CHAI_ENABLE_UM)
#include <cuda_runtime_api.h>
#endif

namespace chai {

template<typename T>
CHAI_INLINE
void* ArrayManager::reallocate(void* pointer, size_t elems, PointerRecord* pointer_record)
{
  ExecutionSpace my_space = CPU;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (pointer_record->m_pointers[space] == pointer) {
      my_space = static_cast<ExecutionSpace>(space);
      break;
    }
  }

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (!pointer_record->m_owned[space]) {
      CHAI_LOG_DEBUG( "Cannot reallocate unowned pointer");
      return pointer_record->m_pointers[my_space];
    }
  }

  // Call callback with ACTION_FREE before changing the size
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (pointer_record->m_pointers[space]) {
       callback(pointer_record, ACTION_FREE, ExecutionSpace(space));
    }
  }

  // Update the pointer record size
  size_t old_size = pointer_record->m_size;
  size_t new_size = sizeof(T) * elems;
  pointer_record->m_size = new_size;

  // only copy however many bytes overlap
  size_t num_bytes_to_copy = std::min(old_size, new_size);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    void* old_ptr = pointer_record->m_pointers[space];

    if (old_ptr) {
      void* new_ptr = m_allocators[space]->allocate(new_size);
      m_resource_manager.copy(new_ptr, old_ptr, num_bytes_to_copy);
      m_allocators[space]->deallocate(old_ptr);

      pointer_record->m_pointers[space] = new_ptr;
      callback(pointer_record, ACTION_ALLOC, ExecutionSpace(space));

      m_pointer_map.erase(old_ptr);
      m_pointer_map.insert(new_ptr, pointer_record);
    }
  }

  return pointer_record->m_pointers[my_space];
}

#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
typename ArrayManager::T_non_const<T> ArrayManager::pick(T* src_ptr, size_t index)
{
  T_non_const<T> val;
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(&val), const_cast<T_non_const<T>*>(src_ptr+index), sizeof(T));
  m_resource_manager.deregisterAllocation(&val);
  return val;
}

template<typename T>
CHAI_INLINE
void ArrayManager::set(T* dst_ptr, size_t index, const T& val)
{
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(dst_ptr+index), const_cast<T_non_const<T>*>(&val), sizeof(T));
  m_resource_manager.deregisterAllocation(const_cast<T_non_const<T>*>(&val));
}
#endif

CHAI_INLINE
void ArrayManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

CHAI_INLINE
umpire::Allocator ArrayManager::getAllocator(ExecutionSpace space) {
   return *m_allocators[space];
}

CHAI_INLINE
void ArrayManager::setAllocator(ExecutionSpace space, umpire::Allocator &allocator) {
   *m_allocators[space] = allocator;
}

CHAI_INLINE
bool ArrayManager::syncIfNeeded() {
  if (!m_synced_since_last_kernel) {
     synchronize();
     m_synced_since_last_kernel = true;
     return true;
  }
  return false;
}
} // end of namespace chai

#endif // CHAI_ArrayManager_INL
