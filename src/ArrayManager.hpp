#ifndef CHAI_ArrayManager_HPP
#define CHAI_ArrayManager_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/PointerRecord.hpp"

#include <map>
#include <set>

namespace chai {

class ArrayManager
{
  public:
  static PointerRecord s_null_record;

  static ArrayManager* getArrayManager();

  void registerHostPointer(void* ptr, size_t size);

  void* getDevicePointer(void* host_ptr);

  void setExecutionSpace(ExecutionSpace space);
  ExecutionSpace getExecutionSpace();

  void* move(void* host_pointer);

  void registerTouch(void* pointer);

  PointerRecord& getPointerRecord(void* host_ptr);

  void move(PointerRecord& record, ExecutionSpace space);

  template<typename T>
  void* allocate(size_t size);

  size_t getSize(void* host_pointer);

  protected:

  ArrayManager();

  private:

  static ArrayManager* s_resource_manager_instance;

  ExecutionSpace m_current_execution_space;

  std::map<void *, PointerRecord> m_pointer_map;
};

}

#endif
