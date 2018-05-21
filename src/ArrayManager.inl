// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
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

#if defined(CHAI_ENABLE_CUDA)
#include "cuda_runtime_api.h"
#endif

#include <iostream>

#include "umpire/ResourceManager.hpp"

namespace chai {

CHAI_INLINE
PointerRecord* ArrayManager::getPointerRecord(void* pointer) 
{
  auto record = m_pointer_map.find(pointer);
  if (record != m_pointer_map.end()) {
    return record->second;
  } else {
    return &s_null_record;
  }
}

CHAI_INLINE
void* ArrayManager::makeManaged(void* pointer, size_t size, ExecutionSpace space, bool owned)
{
  m_resource_manager.registerAllocation(pointer, new umpire::util::AllocationRecord{pointer, size, m_allocators[space]->getAllocationStrategy()});

  registerPointer(pointer, size, space, owned);
  
  auto pointer_record = getPointerRecord(pointer);
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    // If pointer is already active on some execution space, return that pointer
    if(pointer_record->m_touched[i] == true) return pointer_record->m_pointers[i];
  }

  return pointer;
}

template<typename T>
CHAI_INLINE
void* ArrayManager::allocate(size_t elems, ExecutionSpace space, UserCallback const &f)
{
  if (space == NONE) {
    return nullptr;
  }

  void * ret = nullptr;
  ret = m_allocators[space]->allocate(sizeof(T) * elems);

  CHAI_LOG("ArrayManager", "Allocated array at: " << ret);

  registerPointer(ret, sizeof(T) * elems, space);
  
  auto pointer_record = getPointerRecord(ret);
  pointer_record->m_user_callback = f;  
  pointer_record->m_user_callback(ACTION_ALLOC, space, sizeof(T) * elems);
  
  return ret;
}

template<typename T>
CHAI_INLINE
void* ArrayManager::reallocate(void* pointer, size_t elems)
{
  auto pointer_record = getPointerRecord(pointer);

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

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    void* old_ptr = pointer_record->m_pointers[space];

    if (old_ptr) {
      pointer_record->m_user_callback(ACTION_ALLOC, ExecutionSpace(space), sizeof(T) * elems);
      void* new_ptr = m_allocators[space]->allocate(sizeof(T)*elems);

      m_resource_manager.copy(new_ptr, old_ptr, sizeof(T)*elems);

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

CHAI_INLINE
void* ArrayManager::allocate(
    PointerRecord* pointer_record, ExecutionSpace space)
{
  void * ret = nullptr;
  auto size = pointer_record->m_size;
  
  pointer_record->m_user_callback(ACTION_ALLOC, space, size);

  ret = m_allocators[space]->allocate(size);
  registerPointer(ret, pointer_record, space);

  return ret;
}

CHAI_INLINE
void ArrayManager::free(void* pointer)
{
  auto pointer_record = getPointerRecord(pointer);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (pointer_record->m_pointers[space]) {
      if(pointer_record->m_owned[space]) {
        pointer_record->m_user_callback(ACTION_FREE, ExecutionSpace(space), pointer_record->m_size);    
        void* space_ptr = pointer_record->m_pointers[space];
        m_pointer_map.erase(space_ptr);
        m_allocators[space]->deallocate(space_ptr);
        pointer_record->m_pointers[space] = nullptr;
      }
    }
  }

  delete pointer_record;
}


CHAI_INLINE
size_t ArrayManager::getSize(void* ptr)
{
  auto pointer_record = getPointerRecord(ptr);
  return pointer_record->m_size;
}

CHAI_INLINE
void ArrayManager::setDefaultAllocationSpace(ExecutionSpace space)
{
  m_default_allocation_space = space;
}

CHAI_INLINE
ExecutionSpace ArrayManager::getDefaultAllocationSpace()
{
  return m_default_allocation_space;
}


CHAI_INLINE
void ArrayManager::setUserCallback(void *pointer, UserCallback const &f)
{
  auto pointer_record = getPointerRecord(pointer);
  pointer_record->m_user_callback = f;
}


} // end of namespace chai

#endif // CHAI_ArrayManager_INL
