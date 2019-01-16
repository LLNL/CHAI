// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
// rights reserved.
// 
// Produced at the Lawrence Livermore National Laboratory.
// 
// This file is part of CHAI.
// 
// LLNL-CODE-705877
// 
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
// 
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
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
      CHAI_LOG("ArrayManager", "Cannot reallocate unowned pointer");
      return pointer_record->m_pointers[my_space];
    }
  }
  
  // only copy however many bytes overlap
  size_t num_bytes_to_copy = std::min(sizeof(T)*elems, pointer_record->m_size);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    void* old_ptr = pointer_record->m_pointers[space];

    if (old_ptr) {
      pointer_record->m_user_callback(ACTION_ALLOC, ExecutionSpace(space), sizeof(T) * elems);
      void* new_ptr = m_allocators[space]->allocate(sizeof(T)*elems);

      m_resource_manager.copy(new_ptr, old_ptr, num_bytes_to_copy);

      pointer_record->m_user_callback(ACTION_FREE, ExecutionSpace(space), sizeof(T) * elems);
      m_allocators[space]->deallocate(old_ptr);

      pointer_record->m_pointers[space] = new_ptr;

      m_pointer_map.erase(old_ptr);
      m_pointer_map[new_ptr] = pointer_record;
    }
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

CHAI_INLINE
void ArrayManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
