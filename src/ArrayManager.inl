#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"

#if defined(ENABLE_CUDA)
#include "cuda_runtime_api.h"
#endif

#if defined(ENABLE_CNMEM)
#include "cnmem.h"
#endif

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

#if defined(ENABLE_CUDA)
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
#endif

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
#if defined(ENABLE_CUDA)
#if defined(ENABLE_CNMEM)
    cnmemMalloc(&ret, sizeof(T) * elems, NULL);
#else
    cudaMalloc(&ret, sizeof(T) * elems);
#endif
#endif
  }

  CHAI_LOG("ArrayManager", "Allocated array at: " << ret);

  registerPointer(ret, sizeof(T) * elems, space);

  return ret;
}

template<typename T>
CHAI_INLINE
void* ArrayManager::reallocate(void* pointer, size_t elems)
{
  auto pointer_record = getPointerRecord(pointer);

#if defined(ENABLE_CUDA)
  auto space = (pointer_record->m_pointers[CPU] == pointer) ? CPU : GPU; 
#else
  auto space = CPU;
#endif

  if (pointer_record->m_pointers[CPU]) {
    void* old_ptr = pointer_record->m_pointers[CPU];
    void* ptr = ::realloc(old_ptr, sizeof(T) * elems);
    pointer_record->m_pointers[CPU] = ptr;

    m_pointer_map.erase(old_ptr);
    m_pointer_map[ptr] = pointer_record;
  } 
  
#if defined(ENABLE_CUDA)
  if (pointer_record->m_pointers[GPU]) {
    void* old_ptr = pointer_record->m_pointers[GPU];
#if defined(ENABLE_CNMEM)
    void* ptr;
    cnmemMalloc(&ptr, sizeof(T) * elems, NULL);
    cudaMemcpy(ptr, old_ptr, pointer_record->m_size, cudaMemcpyDeviceToDevice);
    cnmemFree(old_ptr, NULL);
#else
    void* ptr;
    cudaMalloc(&ptr, sizeof(T) * elems);
    cudaMemcpy(ptr, old_ptr, pointer_record->m_size, cudaMemcpyDeviceToDevice);
    cudaFree(old_ptr);
#endif
    pointer_record->m_pointers[GPU] = ptr;

    m_pointer_map.erase(old_ptr);
    m_pointer_map[ptr] = pointer_record;
  }
#endif

  pointer_record->m_size = sizeof(T) * elems;
  return pointer_record->m_pointers[space];
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
#if defined(ENABLE_CUDA)
#if defined(ENABLE_CNMEM)
    cnmemMalloc(&ret, size, NULL);
#else
    cudaMalloc(&ret, size);
#endif
#endif
  }

  registerPointer(ret, pointer_record, space);

  return ret;
}

CHAI_INLINE
void ArrayManager::free(void* pointer)
{
  auto pointer_record = getPointerRecord(pointer);

  if (pointer_record->m_pointers[CPU]) {
    void* cpu_ptr = pointer_record->m_pointers[CPU];
    m_pointer_map.erase(cpu_ptr);
    ::free(cpu_ptr);
    pointer_record->m_pointers[CPU] = nullptr;
  } 
  
#if defined(ENABLE_CUDA)
  if (pointer_record->m_pointers[GPU]) {
    void* gpu_ptr = pointer_record->m_pointers[GPU];
    m_pointer_map.erase(gpu_ptr);
#if defined(ENABLE_CNMEM)
    cnmemFree(gpu_ptr, NULL);
#else
    cudaFree(gpu_ptr);
#endif
    pointer_record->m_pointers[GPU] = nullptr;
  }
#endif

  delete pointer_record;
}


CHAI_INLINE
size_t ArrayManager::getSize(void* ptr)
{
  auto pointer_record = getPointerRecord(ptr);
  return pointer_record->m_size;
}

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
