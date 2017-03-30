#include "ArrayManager.hpp"

namespace chai {

ArrayManager* ArrayManager::s_resource_manager_instance = nullptr;
PointerRecord ArrayManager::s_null_record = PointerRecord();

ArrayManager* ArrayManager::getArrayManager() {
  if (!s_resource_manager_instance) {
    s_resource_manager_instance = new ArrayManager();
  }

  return s_resource_manager_instance;
}

ArrayManager::ArrayManager() :
  m_pointer_map()
{
  m_pointer_map.clear();
}

void ArrayManager::registerPointer(void* ptr, size_t size, ExecutionSpace space) {
  auto found_pointer_record = m_pointer_map.find(ptr);

  if (found_pointer_record != m_pointer_map.end()) {
  } else {
    m_pointer_map[ptr] = new PointerRecord();
  }

  auto & pointer_record = m_pointer_map[ptr];

  if (space == CPU) {
    pointer_record->m_host_pointer = ptr;
  } else if (space == GPU) {
    pointer_record->m_device_pointer = ptr;
  }

  pointer_record->m_size = size;
}

void* ArrayManager::getDevicePointer(void* host_pointer)
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

void ArrayManager::setExecutionSpace(ExecutionSpace space) {
  m_current_execution_space = space;
}

void* ArrayManager::move(void* host_pointer) {
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

ExecutionSpace ArrayManager::getExecutionSpace() {
  return m_current_execution_space;
}

void ArrayManager::registerTouch(void* host_pointer) {
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
