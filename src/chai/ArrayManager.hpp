//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ArrayManager_HPP
#define CHAI_ArrayManager_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/PointerRecord.hpp"
#include "chai/Types.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/util/MemoryMap.hpp"

#include "camp/resource.hpp"

namespace chai
{

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

  using PointerMap = umpire::util::MemoryMap<PointerRecord*>;

  static PointerRecord s_null_record;

  /*!
   * \brief Get the singleton instance.
   *
   * \return Pointer to the ArrayManager instance.
   *
   */
  CHAI_HOST_DEVICE
  static ArrayManager* getInstance();

  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  void setExecutionSpace(ExecutionSpace space);
  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  void setExecutionSpace(ExecutionSpace space, camp::resources::Resource *resource);

  /*!
   * \brief Get the current execution space.
   *
   * \return The current execution space.jo
   */
  ExecutionSpace getExecutionSpace();

  
  camp::resources::Resource* getResource();

  /*!
   * \brief Move data in pointer to the current execution space.
   *
   * \param pointer Pointer to data in any execution space.
   * \return Pointer to data in the current execution space.
   */
  void* move(void* pointer,
             PointerRecord* pointer_record,
             ExecutionSpace = NONE);
  void* move(void* pointer,
             PointerRecord* pointer_record,
	     camp::resources::Resource* resource,
             ExecutionSpace = NONE);


  /*!
   * \brief Register a touch of the pointer in the current execution space.
   *
   * \param pointer Raw pointer to register a touch of.
   */
  void registerTouch(PointerRecord* pointer_record);

  /*!
   * \brief Register a touch of the pointer in the given execution space.
   *
   * The pointer doesn't need to exist in the space being touched.
   *
   * \param pointer Raw pointer to register a touch of.
   * \param space Space to register touch.
   */
  void registerTouch(PointerRecord* pointer_record, ExecutionSpace space);

  /*!
   * \brief Make a new allocation of the data described by the PointerRecord in
   * the given space.
   *
   * \param pointer_record
   * \param space Space in which to make the allocation.
   */
  void allocate(PointerRecord* pointer_record, ExecutionSpace space = CPU);

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
  template <typename T>
  void* reallocate(void* pointer, size_t elems, PointerRecord* record);

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
   * \brief Free all allocations associated with the given PointerRecord.
   */
  void free(PointerRecord* pointer);

#if defined(CHAI_ENABLE_PICK)
  template <typename T>
  T_non_const<T> pick(T* src_ptr, size_t index);

  template <typename T>
  void set(T* dst_ptr, size_t index, const T& val);
#endif

  /*!
   * \brief Get the size of the given pointer.
   *
   * \param pointer Pointer to find the size of.
   * \return Size of pointer.
   */
  size_t getSize(void* pointer);

  PointerRecord* makeManaged(void* pointer,
                             size_t size,
                             ExecutionSpace space,
                             bool owned);

  /*!
   * \brief Assign a user-defined callback triggerd upon memory operations.
   */
  void setUserCallback(void* pointer, UserCallback const& f);

  /*!
   * \brief Set touched to false in all spaces for the given PointerRecord.
   *
   * \param pointer_record PointerRecord to reset.
   */
  void resetTouch(PointerRecord* pointer_record);

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
   * \brief Create a copy of the given PointerRecord with a new allocation
   *  in the active space.
   *
   * \param record The PointerRecord to copy.
   *
   * \return A copy of the given PointerRecord, must be free'd with delete.
   */
  PointerRecord* deepCopyRecord(PointerRecord const* record);

  /*!
   * \brief Create a copy of the pointer map.
   *
   * \return A copy of the pointer map. Can be used to find memory leaks.
   */
  std::unordered_map<void*, const PointerRecord*> getPointerMap() const;

  /*!
   * \brief Get the total number of arrays registered with the array manager.
   *
   * \return The total number of arrays registered with the array manager.
   */
  size_t getTotalNumArrays() const;

  /*!
   * \brief Get the total amount of memory allocated.
   *
   * \return The total amount of memory allocated.
   */
  size_t getTotalSize() const;

  int getAllocatorId(ExecutionSpace space) const;

  /*!
   * \brief Turn callbacks on.
   */
  void enableCallbacks() { m_callbacks_active = true; }

  /*!
   * \brief Turn callbacks off.
   */
  void disableCallbacks() { m_callbacks_active = false; }

  /*!
   * \brief Turn on device synchronization after every kernel.
   */
  void enableDeviceSynchronize() { m_device_synchronize = true; }

  /*!
   * \brief Turn off device synchronization after every kernel.
   */
  void disableDeviceSynchronize() { m_device_synchronize = false; }

  /*!
   * \brief Turn on device synchronization after every kernel.
   */
  bool deviceSynchronize() { return m_device_synchronize; }

protected:
  /*!
   * \brief Construct a new ArrayManager.
   *
   * The constructor is a protected member, ensuring that it can
   * only be called by the singleton getInstance method.
   */
  ArrayManager();

private:

  /*!
   * \brief Registering an allocation with the ArrayManager
   *
   * \param record PointerRecord of this allocation.
   * \param space Space in which the pointer was allocated.
   * \param owned Should the allocation be free'd by CHAI?
   */
  void registerPointer(PointerRecord* record,
                       ExecutionSpace space,
                       bool owned = true);

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
  void move(PointerRecord* record, ExecutionSpace space, camp::resources::Resource* resource);

  /*!
   * \brief Execute a user callback if callbacks are active
   *
   * \param record The pointer record containing the callback
   * \param action The event that occurred
   * \param space The space in which the event occurred
   * \param size The number of bytes in the array associated with this pointer record
   */
  inline void callback(PointerRecord* record,
                       Action action,
                       ExecutionSpace space,
                       size_t size) const {
     if (m_callbacks_active && record) {
        record->m_user_callback(action, space, size);
     }
  }

  /*!
   * current execution space.
   */
  ExecutionSpace m_current_execution_space;

  /*!
   * current resource.
   */
  camp::resources::Resource* m_current_resource;


  /**
   * Default space for new allocations
   */
  ExecutionSpace m_default_allocation_space;

  /*!
   * Map of active ManagedArray pointers to their corresponding PointerRecord.
   */
  PointerMap m_pointer_map;

  /*!
   *
   * \brief Array of umpire::Allocators, indexed by ExecutionSpace.
   */
  umpire::Allocator* m_allocators[NUM_EXECUTION_SPACES];

  umpire::ResourceManager& m_resource_manager;

  mutable std::mutex m_mutex;

  /*!
   * \brief Controls whether or not callbacks are called.
   */
  bool m_callbacks_active;

  /*!
   * Whether or not to synchronize on device after every CHAI kernel.
   */
  bool m_device_synchronize = false;
};

}  // end of namespace chai

#include "chai/ArrayManager.inl"

#endif  // CHAI_ArrayManager_HPP
