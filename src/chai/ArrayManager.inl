//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
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
  ExecutionSpace my_space;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (pointer_record->m_pointers[space] == pointer) {
      my_space = static_cast<ExecutionSpace>(space);
    }
  }

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if(!pointer_record->m_owned[space]) {
      CHAI_LOG(Debug, "Cannot reallocate unowned pointer");
      return pointer_record->m_pointers[my_space];
    }
  }

  // only copy however many bytes overlap
  size_t num_bytes_to_copy = std::min(sizeof(T)*elems, pointer_record->m_size);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    void* old_ptr = pointer_record->m_pointers[space];

    pointer_record->m_user_callback(ACTION_ALLOC, ExecutionSpace(space), sizeof(T) * elems);
    void* new_ptr = m_allocators[space]->allocate(sizeof(T)*elems);

    if (old_ptr) {
      m_resource_manager.copy(new_ptr, old_ptr, num_bytes_to_copy);
    }

    pointer_record->m_user_callback(ACTION_FREE, ExecutionSpace(space), sizeof(T) * elems);

    if (old_ptr) {
      m_allocators[space]->deallocate(old_ptr);
    }

    pointer_record->m_pointers[space] = new_ptr;

    if (old_ptr) {
      m_pointer_map.erase(old_ptr);
    }

    m_pointer_map.insert(new_ptr, pointer_record);
  }

  pointer_record->m_size = sizeof(T) * elems;
  return pointer_record->m_pointers[my_space];
}

#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
typename ArrayManager::T_non_const<T> ArrayManager::pick(T* src_ptr, size_t index)
{
  T_non_const<T> val;
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), new umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(&val), const_cast<T_non_const<T>*>(src_ptr+index), sizeof(T));
  m_resource_manager.deregisterAllocation(&val);
  return val;
}

template<typename T>
CHAI_INLINE
void ArrayManager::set(T* dst_ptr, size_t index, const T& val)
{
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), new umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(dst_ptr+index), const_cast<T_non_const<T>*>(&val), sizeof(T));
  m_resource_manager.deregisterAllocation(const_cast<T_non_const<T>*>(&val));
}
#endif

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
