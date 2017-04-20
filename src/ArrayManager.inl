#include "chai/ArrayManager.hpp"

#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

namespace chai {

inline PointerRecord* ArrayManager::getPointerRecord(void* host_ptr) {
  auto record = m_pointer_map.find(host_ptr);
  if (record != m_pointer_map.end()) {
    return record->second;
  } else {
    return &s_null_record;
  }
}

inline void ArrayManager::move(const PointerRecord* record, ExecutionSpace space) {
  if ( space == NONE ) {
    return;
  }

  if (record->m_host_pointer) {
    if (space == GPU && record->m_host_touched) {
      cudaMemcpy(record->m_device_pointer, record->m_host_pointer, 
          record->m_size, cudaMemcpyHostToDevice);
    }

    if (space == CPU && record->m_device_touched) {
      cudaMemcpy(record->m_host_pointer, record->m_device_pointer,
          record->m_size, cudaMemcpyDeviceToHost);
    }
  }
}

template<typename T>
void* ArrayManager::allocate(size_t size, ExecutionSpace space)
{
  void * ret = nullptr;

  if (space == CPU) {
    posix_memalign(static_cast<void **>(&ret), 64, sizeof(T) * size); 
  } else {
    cudaMalloc(&ret, sizeof(T) * size);
  }

  registerPointer(ret, sizeof(T) * size, space);
  return ret;
}

inline size_t ArrayManager::getSize(void* ptr) {
  auto pointer_record = getPointerRecord(ptr);
  return pointer_record->m_size;
}

} // end of namespace chai

#endif // CHAI_ArrayManager_INL
