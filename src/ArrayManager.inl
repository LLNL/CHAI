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

#if defined(CHAI_ENABLE_CNMEM)
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

#if defined(CHAI_ENABLE_CUDA)
  if (space == GPU) {
#if defined(CHAI_ENABLE_UM)
    if (record->m_pointers[UM]) {
      cudaMemPrefetchAsync(record->m_pointers[UM], record->m_size, 0);
    } else {
#endif
      if (!record->m_pointers[GPU]) {
        allocate(record, GPU);
      }
      if (record->m_touched[CPU]) {
        cudaMemcpy(record->m_pointers[GPU], record->m_pointers[CPU], 
            record->m_size, cudaMemcpyHostToDevice);
      }
#if defined(CHAI_ENABLE_UM)
    }
#endif
  }

  if (space == CPU) {
#if defined(CHAI_ENABLE_UM)
    if (record->m_pointers[UM]) {
    } else {
#endif
      if (!record->m_pointers[CPU]) {
        allocate(record, CPU);
      }

      if (record->m_touched[GPU]) {
        cudaMemcpy(record->m_pointers[CPU], record->m_pointers[GPU],
            record->m_size, cudaMemcpyDeviceToHost);
      }
#if defined(CHAI_ENABLE_UM)
    }
#endif
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
#if defined(CHAI_ENABLE_CUDA)
  } else if (space == GPU) {
#if defined(CHAI_ENABLE_CNMEM)
    cnmemMalloc(&ret, sizeof(T) * elems, NULL);
#else
    cudaMalloc(&ret, sizeof(T) * elems);
#endif
#if defined(CHAI_ENABLE_UM)
  } else if (space == UM) {
    cudaMallocManaged(&ret, sizeof(T) * elems);
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

#if defined(CHAI_ENABLE_CUDA)
  auto space = (pointer_record->m_pointers[CPU] == pointer) ? CPU :
    (pointer_record->m_pointers[GPU] == pointer) ? GPU : 
#if defined(CHAI_ENABLE_UM)
    UM;
#else
  GPU;
#endif
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
  
#if defined(CHAI_ENABLE_CUDA)
  if (pointer_record->m_pointers[GPU]) {
    void* old_ptr = pointer_record->m_pointers[GPU];
#if defined(CHAI_ENABLE_CNMEM)
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

#if defined(CHAI_ENABLE_UM)
  if (pointer_record->m_pointers[UM]) {
    void* old_ptr = pointer_record->m_pointers[GPU];
    void* ptr;
    cudaMallocManaged(&ptr, sizeof(T) * elems);
    cudaMemcpy(ptr, old_ptr, pointer_record->m_size, cudaMemcpyDefault);
    cudaFree(old_ptr);
    pointer_record->m_pointers[UM] = ptr;

    m_pointer_map.erase(old_ptr);
    m_pointer_map[ptr] = pointer_record;
  }
#endif
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
#if defined(CHAI_ENABLE_CUDA)
  } else if (space == GPU) {
#if defined(CHAI_ENABLE_CNMEM)
    cnmemMalloc(&ret, size, NULL);
#else
    cudaMalloc(&ret, size);
#endif
#if defined(CHAI_ENABLE_UM)
  } else if (space == UM) {
    cudaMallocManaged(&ret, size);
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
  
#if defined(CHAI_ENABLE_CUDA)
  if (pointer_record->m_pointers[GPU]) {
    void* gpu_ptr = pointer_record->m_pointers[GPU];
    m_pointer_map.erase(gpu_ptr);
#if defined(CHAI_ENABLE_CNMEM)
    cnmemFree(gpu_ptr, NULL);
#else
    cudaFree(gpu_ptr);
#endif
    pointer_record->m_pointers[GPU] = nullptr;
  }

#if defined(CHAI_ENABLE_UM)
  if (pointer_record->m_pointers[UM]) {
    void* um_ptr = pointer_record->m_pointers[UM];
    m_pointer_map.erase(um_ptr);
    cudaFree(um_ptr);
    pointer_record->m_pointers[UM] = nullptr;
  }
#endif
#endif

  delete pointer_record;
}

#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
void ArrayManager::transferValue(T* pointer, T& val, size_t index, Direction tofrom)
{
#if defined(CHAI_ENABLE_CUDA)
  auto pointer_record = getPointerRecord(pointer);
  // if active on non-HOST execution space (GPU)
  //TODO: Extend this to n arbitrary execution spaces
  if(pointer_record->m_touched[GPU]) {
    //FIXME: "pointer" should work just fine since it is THE active pointer
    T* registered_ptr = (T*)(pointer_record->m_pointers[GPU]);
    if(index*sizeof(T) < pointer_record->m_size) {
      cudaError_t err;
      if(tofrom == Direction::DeviceToHost) err = cudaMemcpy(&val, registered_ptr+index, sizeof(T), cudaMemcpyDeviceToHost); //device to host
      else err = cudaMemcpy(registered_ptr+index, &val, sizeof(T), cudaMemcpyHostToDevice); //host to device
    }
    // else {} //TODO: How to fail
  } else {
#endif
    if(tofrom == Direction::DeviceToHost) val = pointer[index]; //device to host
    else pointer[index] = val; //host to device
#if defined(CHAI_ENABLE_CUDA)
  }
#endif
}

template<typename T>
CHAI_INLINE
void ArrayManager::pick(T* pointer, size_t index, T_non_const<T>& val)
{
  transferValue((T_non_const<T>*)pointer, val, index, Direction::DeviceToHost);
}

template<typename T>
CHAI_INLINE
void ArrayManager::set(T* pointer, size_t index, T& val)
{
  transferValue(pointer, val, index, Direction::HostToDevice);
}

template<typename T>
CHAI_INLINE
void ArrayManager::modifyValue(T* pointer, size_t index, T val)
{
#if defined(CHAI_ENABLE_CUDA)
  auto pointer_record = getPointerRecord(pointer);
  // if active on non-HOST execution space (GPU)
  //TODO: Extend this to n arbitrary execution spaces
  if(pointer_record->m_touched[GPU]) {
    //FIXME: pointer should work just fine since it is THE active pointer
    T* registered_ptr = (T*)(pointer_record->m_pointers[GPU]);
    if(index*sizeof(T) < pointer_record->m_size) {
      T temp;
      cudaError_t err;
      err = cudaMemcpy(&temp, registered_ptr+index, sizeof(T), cudaMemcpyDeviceToHost); //device to host
      temp = temp + val;
      err = cudaMemcpy(registered_ptr+index, &temp, sizeof(T), cudaMemcpyHostToDevice); //host to device
    }
    // else {} //TODO: How to fail
  } else {
#endif
    pointer[index] = pointer[index] + val;
#if defined(CHAI_ENABLE_CUDA)
  }
#endif
}

template<typename T>
CHAI_INLINE
void ArrayManager::incr(T* pointer, size_t index)
{
  modifyValue(pointer, index, (T)1);
}

template<typename T>
CHAI_INLINE
void ArrayManager::decr(T* pointer, size_t index)
{
  modifyValue(pointer, index, (T)-1);
}
#endif

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

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
