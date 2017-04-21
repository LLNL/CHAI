#include "chai/ArrayManager.hpp"

#include "chai/ChaiMacros.hpp"

#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

#include <iostream>

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
void ArrayManager::move(PointerRecord* record, ExecutionSpace space) 
{
  if ( space == NONE ) {
    return;
  }

  if (space == GPU) {
    if (!record->m_pointers[GPU]) {
      allocate(record, GPU);
    }
    if (record->m_touched[CPU]) {
      cudaMemcpy(record->m_pointers[GPU], record->m_pointers[CPU], 
          record->m_size, cudaMemcpyHostToDevice);
    }
  }

  if (space == CPU) {
    if (!record->m_pointers[CPU]) {
      allocate(record, CPU);
    }

    if (record->m_touched[GPU]) {
      cudaMemcpy(record->m_pointers[CPU], record->m_pointers[GPU],
          record->m_size, cudaMemcpyDeviceToHost);
    }
  }

  resetTouch(record);
}

template<typename T>
CHAI_INLINE
void* ArrayManager::allocate(size_t elems, ExecutionSpace space)
{
  void * ret = nullptr;

  if (space == CPU) {
    posix_memalign(static_cast<void **>(&ret), 64, sizeof(T) * elems); 
  } else {
    cudaMalloc(&ret, sizeof(T) * elems);
  }

  registerPointer(ret, sizeof(T) * elems, space);

#ifdef DEBUG
  std::cout << "[ArrayManager] Allocated array at: " << ret << std::endl;
#endif

  return ret;
}

CHAI_INLINE
void* ArrayManager::allocate(
    PointerRecord* pointer_record, ExecutionSpace space)
{
  void * ret = nullptr;
  auto size = pointer_record->m_size;

  if (space == CPU) {
    posix_memalign(static_cast<void **>(&ret), 64, size); 
  } else {
    cudaMalloc(&ret, size);
  }

  registerPointer(ret, pointer_record, space);

  return ret;
}

CHAI_INLINE
CHAI_HOST void free(void* pointer)
{
  auto pointer_record = getPointerRecord(pointer);

  if (pointer_record->m_pointer[CPU]) {
    ::free(&pointer_record->m_pointer[i]);
  } 
  
  if (pointer_record->m_pointer[GPU]) {
    cudaFree(&pointer_record->m_pointer[GPU]);
  }
}


CHAI_INLINE
size_t ArrayManager::getSize(void* ptr)
{
  auto pointer_record = getPointerRecord(ptr);
  return pointer_record->m_size;
}

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
