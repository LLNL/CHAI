//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPtrManager_INL
#define CHAI_SharedPtrManager_INL

#include "chai/config.hpp"

#include "chai/SharedPtrManager.hpp"
#include "chai/ChaiMacros.hpp"

#include <iostream>

#include "umpire/ResourceManager.hpp"

#include <cuda_runtime_api.h>

//#if defined(CHAI_ENABLE_UM)
//#if !defined(CHAI_THIN_GPU_ALLOCATE)
//#include <cuda_runtime_api.h>
//#endif
//#endif

namespace chai {

//template<typename T>
//CHAI_INLINE
//void* SharedPtrManager::reallocate(void* pointer, size_t elems, msp_pointer_record* pointer_record)
//{
//  ExecutionSpace my_space = CPU;
//
//  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
//    if (pointer_record->m_pointers[space] == pointer) {
//      my_space = static_cast<ExecutionSpace>(space);
//      break;
//    }
//  }
//
//  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
//    if (!pointer_record->m_owned[space]) {
//      CHAI_LOG(Debug, "Cannot reallocate unowned pointer");
//      return pointer_record->m_pointers[my_space];
//    }
//  }
//
//  // Call callback with ACTION_FREE before changing the size
//  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
//    if (pointer_record->m_pointers[space]) {
//       callback(pointer_record, ACTION_FREE, ExecutionSpace(space));
//    }
//  }
//
//  // Update the pointer record size
//  size_t old_size = pointer_record->m_size;
//  size_t new_size = sizeof(T) * elems;
//  pointer_record->m_size = new_size;
//
//  // only copy however many bytes overlap
//  size_t num_bytes_to_copy = std::min(old_size, new_size);
//
//  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
//    void* old_ptr = pointer_record->m_pointers[space];
//
//    if (old_ptr) {
//      void* new_ptr = m_allocators[space]->allocate(new_size);
//      m_resource_manager.copy(new_ptr, old_ptr, num_bytes_to_copy);
//      m_allocators[space]->deallocate(old_ptr);
//
//      pointer_record->m_pointers[space] = new_ptr;
//      callback(pointer_record, ACTION_ALLOC, ExecutionSpace(space));
//
//      m_pointer_map.erase(old_ptr);
//      m_pointer_map.insert(new_ptr, pointer_record);
//    }
//  }
//
//  return pointer_record->m_pointers[my_space];
//}
template<typename Ptr>
msp_pointer_record* SharedPtrManager::makeSharedPtrRecord(std::initializer_list<Ptr*> pointers,
                                                          std::initializer_list<chai::ExecutionSpace> spaces,
                                                          size_t size,
                                                          bool owned)
{
  int i = 0;
  for (Ptr* ptr : pointers) {
    if (ptr == nullptr) return &s_null_record;
    m_resource_manager.registerAllocation(ptr, 
        {ptr, size, m_allocators[spaces.begin()[i++]]->getAllocationStrategy()}
    );
  }

  Ptr* lookup_pointer = const_cast<Ptr*>(pointers.begin()[0]);

  auto pointer_record = getPointerRecord(lookup_pointer);

  if (pointer_record == &s_null_record) {
     if (lookup_pointer) {
        pointer_record = new msp_pointer_record();
     } else {
        return pointer_record;
     }
  }
  else {
     CHAI_LOG(Warning, "SharedPtrManager::makeManaged found abandoned pointer record!!!");
     //callback(pointer_record, ACTION_FOUND_ABANDONED, space);
  }

  i=0;
  for (void const* c_ptr : pointers) {
    void* ptr = const_cast<void*>(c_ptr);
    chai::ExecutionSpace space = spaces.begin()[i];

    pointer_record->m_pointers[space] = ptr;
    pointer_record->m_owned[space] = owned;
    registerPointer(pointer_record, space, owned);

    i++;
  }

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_allocators[space] = getAllocatorId(ExecutionSpace(space));
  }

  return pointer_record;
}


#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
typename SharedPtrManager::T_non_const<T> SharedPtrManager::pick(T* src_ptr, size_t index)
{
  T_non_const<T> val;
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(&val), const_cast<T_non_const<T>*>(src_ptr+index), sizeof(T));
  m_resource_manager.deregisterAllocation(&val);
  return val;
}

template<typename T>
CHAI_INLINE
void SharedPtrManager::set(T* dst_ptr, size_t index, const T& val)
{
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(dst_ptr+index), const_cast<T_non_const<T>*>(&val), sizeof(T));
  m_resource_manager.deregisterAllocation(const_cast<T_non_const<T>*>(&val));
}
#endif

CHAI_INLINE
void SharedPtrManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

CHAI_INLINE
umpire::Allocator SharedPtrManager::getAllocator(ExecutionSpace space) {
   return *m_allocators[space];
}

CHAI_INLINE
void SharedPtrManager::setAllocator(ExecutionSpace space, umpire::Allocator &allocator) {
   *m_allocators[space] = allocator;
}

CHAI_INLINE
bool SharedPtrManager::syncIfNeeded() {
  if (!m_synced_since_last_kernel) {
     synchronize();
     m_synced_since_last_kernel = true;
     return true;
  }
  return false;
}
} // end of namespace chai

#endif // CHAI_SharedPtrManager_INL
