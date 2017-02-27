#include "chai/ResourceManager.hpp"

#ifndef CHAI_ResourceManager_INL
#define CHAI_ResourceManager_INL

namespace chai {

inline PointerRecord& ResourceManager::getPointerRecord(void* host_ptr) {
  auto record = m_pointer_map.find(host_ptr);
  if (record != m_pointer_map.end()) {
    return record->second;
  } else {
    return s_null_record;
  }
}

inline void ResourceManager::move(PointerRecord& record, ExecutionSpace space) {
  if ( space == NONE ) {
    return;
  }

  if (record.m_host_pointer) {
    if (space == GPU && record.m_host_touched) {
      cudaMemcpy(record.m_device_pointer, record.m_host_pointer, 
          record.m_size, cudaMemcpyHostToDevice);
    }

    if (space == CPU && record.m_device_touched) {
      cudaMemcpy(record.m_host_pointer, record.m_device_pointer,
          record.m_size, cudaMemcpyDeviceToHost);
    }
  }
}

template<typename T>
void* ResourceManager::allocate(size_t size)
{
  void * ret = nullptr;

  posix_memalign(static_cast<void **>(&ret), 64, sizeof(T) * size); 
  registerHostPointer(ret, sizeof(T) * size);

  return ret;
}

inline size_t ResourceManager::getSize(void* host_pointer) {
  auto & pointer_record = getPointerRecord(host_pointer);
  return pointer_record.m_size;
}

} // end of namespace chai

#endif
