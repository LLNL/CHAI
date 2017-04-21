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

#ifdef DEBUG
  std::cout << "[ArrayManager] Registering " << ptr << " in space " << space << std::endl;
#endif

  pointer_record->m_pointers[space] = ptr;
  pointer_record->m_size = size;
}

void ArrayManager::registerPointer(void* ptr, PointerRecord* record, ExecutionSpace space) 
{

#ifdef DEBUG
  std::cout << "[ArrayManager] Registering " << ptr << " in space" << space << std::endl;
#endif

  record->m_pointers[space] = ptr;
}

void ArrayManager::setExecutionSpace(ExecutionSpace space) {
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
#ifdef DEBUG
  std::cout << "[ArrayManager] " << host_pointer << " touched in space ";
  std::cout << m_current_execution_space << std::endl;
#endif

  auto pointer_record = getPointerRecord(host_pointer);
  pointer_record->m_touched[m_current_execution_space] = true;
}

void ArrayManager::resetTouch(PointerRecord* pointer_record) {
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    pointer_record->m_touched[i] = false;
  }
}

} // end of namespace chai
