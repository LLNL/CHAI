#include "ResourceManager.hpp"

namespace chai {

ResourceManager* ResourceManager::s_resource_manager_instance = nullptr;
PointerRecord ResourceManager::s_null_record = PointerRecord();

ResourceManager* ResourceManager::getResourceManager() {
  if (!s_resource_manager_instance) {
    s_resource_manager_instance = new ResourceManager();
  }

  return s_resource_manager_instance;
}

ResourceManager::ResourceManager() :
  m_pointer_map(),
  m_accessed_variables()
{
  m_pointer_map.clear();
  m_accessed_variables.clear();
}

void ResourceManager::registerHostPointer(void* ptr, size_t size) {
  auto found_pointer_record = m_pointer_map.find(ptr);

  if (found_pointer_record != m_pointer_map.end()) {
    
  } else {
    m_pointer_map[ptr] = PointerRecord();
  }

  auto & pointer_record = m_pointer_map[ptr];

  pointer_record.m_host_pointer = ptr;
  pointer_record.m_size = size;
}

void* ResourceManager::getDevicePointer(void* host_pointer)
{
  if (host_pointer == nullptr) {
    return nullptr;
  }

  auto& pointer_record = getPointerRecord(host_pointer);

  if (pointer_record.m_device_pointer != nullptr) {
    return pointer_record.m_device_pointer;
  } else {
    size_t size = pointer_record.m_size;
    void* device_pointer;

    cudaMalloc(&device_pointer, size);

    pointer_record.m_device_pointer = device_pointer;
    return device_pointer;
  }
}

void ResourceManager::setExecutionSpace(ExecutionSpace space) {
  m_current_execution_space = space;
}

void* ResourceManager::move(void* host_pointer) {
  if (m_current_execution_space == NONE) {
    return nullptr;
  }

  if (m_current_execution_space == GPU) {
    getDevicePointer(host_pointer);
  } 

  auto & pointer_record = getPointerRecord(host_pointer);
  if (pointer_record.m_device_pointer) {
    move(pointer_record, m_current_execution_space);
  }

  return pointer_record.m_device_pointer;
}

ExecutionSpace ResourceManager::getExecutionSpace() {
  return m_current_execution_space;
}

void ResourceManager::registerTouch(void* host_pointer) {
  auto & pointer_record = getPointerRecord(host_pointer);

  if (m_current_execution_space == CPU) {
    pointer_record.m_device_touched = false;
    pointer_record.m_host_touched = true;
  } else if (m_current_execution_space == GPU) {
    pointer_record.m_device_touched = true;
    pointer_record.m_host_touched = false;
  }
}

}
