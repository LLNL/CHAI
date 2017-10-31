// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
// rights reserved.
// 
// Produced at the Lawrence Livermore National Laboratory.
// 
// This file is part of CHAI.
// 
// LLNL-CODE-705877
// 
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
// 
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
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

  template <typename T>
  using T_non_const = typename std::remove_const<T>::type;

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
   * \brief Register a touch of the pointer in the given execution space.
   *
   * The pointer doesn't need to exist in the space being touched.
   *
   * \param pointer Raw pointer to register a touch of.
   * \param space Space to register touch.
   */
  void registerTouch(void* pointer, ExecutionSpace space);

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

  /*!
   * \brief Reallocate data.
   *
   * Data is reallocated in all spaces this pointer is associated with.
   *
   * \param ptr Pointer to address to reallocate
   * \param elems The number of elements to allocate.
   * \tparam T The type of data to allocate.
   * 
   * \return Pointer to the allocated memory.
   */
  template<typename T>
  void* reallocate(void* pointer, size_t elems);

  /*!
   * \brief Set the default space for new ManagedArray allocations.
   *
   * ManagedArrays allocated without an explicit ExecutionSpace argument will
   * be allocated in space after this routine is called.
   *
   * \param space New space for default allocations.
   */
  void setDefaultAllocationSpace(ExecutionSpace space);

  /*!
   * \brief Get the currently set default allocation space.
   *
   * See also setDefaultAllocationSpace.
   *
   * \return Current default space for allocations.
   */
  ExecutionSpace getDefaultAllocationSpace();

  /*!
   * \brief Free all allocations associated with the given raw pointer.
   */
  void free(void* pointer);

#if defined(CHAI_ENABLE_PICK)
  /*!
   * \brief Pick a data value from the ManagedArray.
   *
   * The data value is picked from the current active space 
   * into the host memory
   *
   * \param pointer The active pointer
   * \param index The index of the element to be fetched
   * \param val Destination location of the picked value
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void pick(T* pointer, size_t index, T_non_const<T>& val);

  /*!
   * \brief Set a data value in the ManagedArray.
   *
   * The data value in the ManagedArray is set from
   * the host memory to current active space
   *
   * \param pointer The current active pointer
   * \param index The index of the element to be set 
   * \param val Source location of the value
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void set(T* pointer, size_t index, T& val);

  /*!
   * \brief Increment a data value in the ManagedArray.
   *
   * \param pointer The current active pointer
   * \param index The index of the element to be incremented 
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void incr(T* pointer, size_t index);

  /*!
   * \brief Decrement a data value in the ManagedArray.
   *
   * \param pointer The current active pointer
   * \param index The index of the element to be decremented 
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void decr(T* pointer, size_t index);
#endif

  /*!
   * \brief Get the size of the given pointer.
   *
   * \param pointer Pointer to find the size of.
   * \return Size of pointer.
   */
  size_t getSize(void* pointer);

  protected:

  /*!
   * \brief Construct a new ArrayManager.
   *
   * The constructor is a protected member, ensuring that it can
   * only be called by the singleton getInstance method.
   */
  ArrayManager();

  private:
#if defined(CHAI_ENABLE_PICK)
  enum Direction {
    HostToDevice,
    DeviceToHost
  };

  /*!
   * \brief Increment a data value in the ManagedArray.
   *
   * \param pointer The current active pointer
   * \param index The index of the element to be incremented 
   * \param val The value of the increment (if < 0, the operation is a decrement) 
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void modifyValue(T* pointer, size_t index, T val);

  /*!
   * \brief Transfer a single value to/from the ManagedArray.
   *
   * \param pointer The active pointer
   * \param index The index of the element to be set 
   * \param val Reference to the value
   * \param tofrom Direction of data transfer (DTOH or HTOD)
   * \tparam T The type of data value in ManagedArray.
   * 
   */
  template<typename T>
  void transferValue(T* pointer, T& val, size_t index, Direction tofrom);
#endif


  /*!
   * \brief Make a new allocation of the data described by the PointerRecord in
   * the given space.
   *
   * \param pointer_record 
   * \param space Space in which to make the allocation.
   */
  void* allocate(PointerRecord* pointer_record, ExecutionSpace space=CPU);

  /*!
   * \brief Registering adding a new allocation to an existing PointerRecord.
   *
   * \param pointer Pointer to the allocation to register.
   * \param record PointerRecord which this allocation is added to.
   * \param space Space in which the pointer was allocated.
   */
  void registerPointer(
      void* pointer, PointerRecord* record, ExecutionSpace space);

  /*!
   * \brief Register a new allocation with the ArrayManager.
   *
   * \param pointer Pointer to the allocation to register.
   * \param size Size of the allocation in bytes.
   * \param space Space in which the pointer was allocated.
   */
  void registerPointer(void* pointer, size_t size, ExecutionSpace space=CPU);

  /*!
   * \brief Deregister a PointerRecord from the ArrayManager.
   */
  void deregisterPointer(PointerRecord* record);

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

  /**
   * Default space for new allocations
   */
  ExecutionSpace m_default_allocation_space;

  /*!
   * Map of active ManagedArray pointers to their corresponding PointerRecord.
   */
  std::unordered_map<void *, PointerRecord*> m_pointer_map;
};

} // end of namespace chai

#include "chai/ArrayManager.inl"

#endif // CHAI_ArrayManager_HPP
