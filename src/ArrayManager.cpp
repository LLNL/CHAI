#include "chai/ArrayManager.hpp"

namespace chai {

ArrayManager* ArrayManager::s_resource_manager_instance = nullptr;
PointerRecord ArrayManager::s_null_record = PointerRecord();

ArrayManager* ArrayManager::getInstance() {
  if (!s_resource_manager_instance) {
    s_resource_manager_instance = new ArrayManager();
  }

  return s_resource_manager_instance;
}

ArrayManager::ArrayManager() :
  m_pointer_map()
{
  m_pointer_map.clear();
  m_current_execution_space = NONE;
}

void ArrayManager::registerPointer(void* pointer, size_t size, ExecutionSpace space) {
  CHAI_LOG("ArrayManager", "Registering " << pointer << " in space " << space);

  auto found_pointer_record = m_pointer_map.find(pointer);

  if (found_pointer_record != m_pointer_map.end()) {
  } else {
    m_pointer_map[pointer] = new PointerRecord();
  }

  auto & pointer_record = m_pointer_map[pointer];


  pointer_record->m_pointers[space] = pointer;
  pointer_record->m_size = size;
}

void ArrayManager::registerPointer(void* pointer, PointerRecord* record, ExecutionSpace space) 
{
  CHAI_LOG("ArrayManager", "Registering " << pointer << " in space " << space);

  record->m_pointers[space] = pointer;
}

void ArrayManager::deregisterPointer(PointerRecord* record)
{
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    if (record->m_pointers[i])
      m_pointer_map.erase(record->m_pointers[i]);
  }

  delete record;
}

void ArrayManager::setExecutionSpace(ExecutionSpace space) {
  CHAI_LOG("ArrayManager", "Setting execution space to " << space);

  m_current_execution_space = space;
}

void* ArrayManager::move(void* pointer) {
  if (m_current_execution_space == NONE) {
    return pointer;
  }

  auto pointer_record = getPointerRecord(pointer);
  move(pointer_record, m_current_execution_space);

  return pointer_record->m_pointers[m_current_execution_space];
}

ExecutionSpace ArrayManager::getExecutionSpace() {
  return m_current_execution_space;
}

void ArrayManager::registerTouch(void* host_pointer) {
  CHAI_LOG("ArrayManager", host_pointer << " touched in space " << m_current_execution_space);

  auto pointer_record = getPointerRecord(host_pointer);
  pointer_record->m_touched[m_current_execution_space] = true;
}

void ArrayManager::resetTouch(PointerRecord* pointer_record) {
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    pointer_record->m_touched[i] = false;
  }
}

} // end of namespace chai
