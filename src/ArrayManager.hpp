#ifndef CHAI_ArrayManager_HPP
#define CHAI_ArrayManager_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/PointerRecord.hpp"

#include <unordered_map>

namespace chai {

/*
 * \brief Singleton that manages caching and movement of ManagedArray objects.
 */
class ArrayManager
{
  public:

  static PointerRecord s_null_record;

  /**
   * \brief Get the singleton instance.
   *
   */ 
  static ArrayManager* getArrayManager();

  void* getDevicePointer(void* host_ptr);

  void setExecutionSpace(ExecutionSpace space);
  ExecutionSpace getExecutionSpace();

  void* move(void* host_pointer);

  void registerTouch(void* pointer);

  PointerRecord* getPointerRecord(void* host_ptr);

  void move(const PointerRecord* record, ExecutionSpace space);

  template<typename T>
  void* allocate(size_t size, ExecutionSpace=CPU);

  size_t getSize(void* host_pointer);

  protected:

  /*
   * \brief Constructor.
   */
  ArrayManager();

  private:

  /*
   * \brief Register a new allocation with the ArrayManager.
   */
  void registerPointer(void* ptr, size_t size, ExecutionSpace space=CPU);

  /*
   * \brief Pointer to singleton instance.
   */
  static ArrayManager* s_resource_manager_instance;

  /*
   * \brief Current execution space.
   */
  ExecutionSpace m_current_execution_space;

  /*
   * \brief Map of active ManagedArray pointers to their corresponding
   * PointerRecord.
   */
  std::unordered_map<void *, PointerRecord*> m_pointer_map;
};

} // end of namespace chai

#include "chai/ArrayManager.inl"

#endif // CHAI_ArrayManager_HPP
