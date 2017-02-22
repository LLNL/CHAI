#ifndef CHAI_ResourceManager_HPP
#define CHAI_ResourceManager_HPP

#include "chai/ExecutionSpaces.hpp"

#include <map>
#include <set>

namespace chai {

struct PointerRecord 
{
  void * m_host_pointer;
  void * m_device_pointer;

  size_t m_size;

  bool m_host_touched;
  bool m_device_touched;
};


class ResourceManager
{
  public:
  static PointerRecord s_null_record;

  static ResourceManager* getResourceManager();

  void registerHostPointer(void* ptr, size_t size);

  void* getDevicePointer(void* host_ptr);

  void setExecutionSpace(ExecutionSpace space);
  ExecutionSpace getExecutionSpace();

  void* move(void* host_pointer);

  void registerTouch(void* pointer);

  PointerRecord& getPointerRecord(void* host_ptr) {
    auto record = m_pointer_map.find(host_ptr);
    if (record != m_pointer_map.end()) {
      return record->second;
    } else {
      return s_null_record;
    }
  }

  void move(PointerRecord& record, ExecutionSpace space) {
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
  void* allocate(size_t size)
  {
    void * ret = nullptr;

    posix_memalign(static_cast<void **>(&ret), 64, sizeof(T) * size); 
    registerHostPointer(ret, sizeof(T) * size);

    return ret;
  }

  size_t getSize(void* host_pointer) {
    auto & pointer_record = getPointerRecord(host_pointer);
    return pointer_record.m_size;
  }

  protected:

  ResourceManager();

  private:

  static ResourceManager* s_resource_manager_instance;

  ExecutionSpace m_current_execution_space;

  std::map<void *, PointerRecord> m_pointer_map;
};

}

#endif
