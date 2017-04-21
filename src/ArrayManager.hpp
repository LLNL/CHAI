#ifndef CHAI_ArrayManager_HPP
#define CHAI_ArrayManager_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/PointerRecord.hpp"

#include <unordered_map>

namespace chai {

/*!
 * \brief Singleton that manages caching and movement of ManagedArray objects.
 *
 * The ArrayManager class co-ordinates the allocation and movement of
 * ManagedArray objects. These objects are cached, and data is only copied
 * between ExecutionSpaces when necessary. This functionality is typically
 * hidden behind a programming model layer, such as RAJA, or the exmaple
 * included in util/forall.hpp
 *
 * The ArrayManager is a singleton, so must always be accessed through the
 * static getInstance method. Here is an example using the ArrayManager:
 *
 * \code
 * const chai::ArrayManager* rm = chai::ArrayManager::getInstance();
 * rm->setExecutionSpace(chai::CPU);
 * // Do something in with ManagedArrays on the CPU... but they must be copied!
 * rm->setExecutionSpace(chai::NONE);
 * \endcode
 */
class ArrayManager
{
  public:

  static PointerRecord s_null_record;

  /*!
   * \brief Get the singleton instance.
   *
   * \return Pointer to the ArrayManager instance.
   *
   */ 
  static ArrayManager* getInstance();

  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  void setExecutionSpace(ExecutionSpace space);

  /*!
   * \brief Get the current execution space.
   *
   * \return The current execution space.jo
   */
  ExecutionSpace getExecutionSpace();

  /*!
   * \brief Move data in pointer to the current execution space. 
   *
   * \param pointer Pointer to data in any execution space.
   * \return Pointer to data in the current execution space.
   */
  void* move(void* pointer);

  /*!
   * \brief Register a touch of the pointer in the current execution space.
   *
   * \param pointer Raw pointer to register a touch of.
   */
  void registerTouch(void* pointer);

  /*!
   * \brief Allocate data in the specified space.
   *
   * \param elems The number of elements to allocate.
   * \param space The space in which to allocate the data.
   * \tparam T The type of data to allocate.
   * 
   * \return Pointer to the allocated memory.
   */
  template<typename T>
  void* allocate(size_t elems, ExecutionSpace space=CPU);

  size_t getSize(void* host_pointer);

  protected:

  /*!
   * \brief Construct a new ArrayManager.
   *
   * The constructor is a protected member, ensuring that it can
   * only be called by the singleton getInstance method.
   */
  ArrayManager();

  private:

  /*
   * \brief Make a new allocation of the data described by the PointerRecord in
   * the given space.
   *
   * \param pointer_record 
   * \param space Space in which to make the allocation.
   */
  void* allocate(PointerRecord* pointer_record, ExecutionSpace space=CPU);

  /*
   * \brief Registering adding a new allocation to an existing PointerRecord.
   *
   * \param pointer Pointer to the allocation to register.
   * \param record PointerRecord which this allocation is added to.
   * \param space Space in which the pointer was allocated.
   */
  void registerPointer(
      void* pointer, PointerRecord* record, ExecutionSpace space);

  /*
   * \brief Register a new allocation with the ArrayManager.
   *
   * \param pointer Pointer to the allocation to register.
   * \param size Size of the allocation in bytes.
   * \param space Space in which the pointer was allocated.
   */
  void registerPointer(void* pointer, size_t size, ExecutionSpace space=CPU);

  /*!
   * \brief Move data in PointerRecord to the corresponding ExecutionSpace.
   *
   * \param record
   * \param space
   */
  void move(PointerRecord* record, ExecutionSpace space);

  /*!
   * \brief Find the PointerRecord corresponding to the raw pointer.
   *
   * \param pointer Raw pointer to find the PointerRecord for.
   *
   * \return PointerRecord containing the raw pointer, or an empty
   *         PointerRecord if none found.
   */
  PointerRecord* getPointerRecord(void* pointer);

  /*!
   * \brief Set touched to false in all spaces for the given PointerRecord.
   *
   * \param pointer_record PointerRecord to reset.
   */
  void resetTouch(PointerRecord* pointer_record);

  /*!
   * Pointer to singleton instance.
   */
  static ArrayManager* s_resource_manager_instance;

  /*!
   * Current execution space.
   */
  ExecutionSpace m_current_execution_space;

  /*!
   * Map of active ManagedArray pointers to their corresponding PointerRecord.
   */
  std::unordered_map<void *, PointerRecord*> m_pointer_map;
};

} // end of namespace chai

#include "chai/ArrayManager.inl"

#endif // CHAI_ArrayManager_HPP
