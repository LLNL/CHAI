//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
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

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif
#if defined(CHAI_ENABLE_HIP)
#include "hip/hip_runtime_api.h"
#endif

namespace chai
{
// CHAI_GPU_ERROR_CHECK macro
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

#ifdef CHAI_ENABLE_GPU_ERROR_CHECKING

#ifdef CHAI_ENABLE_CUDA
inline void gpuErrorCheck(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "[CHAI] GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}
#elif defined(CHAI_ENABLE_HIP)
inline void gpuErrorCheck(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) {
      fprintf(stderr, "[CHAI] GPU Error: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}
#endif


#define CHAI_GPU_ERROR_CHECK(code) { gpuErrorCheck((code), __FILE__, __LINE__); }
#else // CHAI_ENABLE_GPU_ERROR_CHECKING
#define CHAI_GPU_ERROR_CHECK(code) code
#endif // CHAI_ENABLE_GPU_ERROR_CHECKING

#endif

// wrapper for hip/cuda synchronize
inline void synchronize() {
#if defined (CHAI_ENABLE_HIP) &&!defined(__HIP_DEVICE_COMPILE__)
   CHAI_GPU_ERROR_CHECK(hipDeviceSynchronize());
#elif defined (CHAI_ENABLE_CUDA) &&!defined(__CUDA_ARCH__)
   CHAI_GPU_ERROR_CHECK(cudaDeviceSynchronize());
#endif
}

#if defined(CHAI_GPUCC)

// wrapper for hip/cuda free
CHAI_HOST inline void gpuFree(void* buffer) {
#if defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipFree(buffer));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaFree(buffer));
#endif
}

// wrapper for hip/cuda malloc
CHAI_HOST inline void gpuMalloc(void** devPtr, size_t size) {
#if defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMalloc(devPtr, size));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMalloc(devPtr, size));
#endif
}

// wrapper for hip/cuda managed malloc
CHAI_HOST inline void gpuMallocManaged(void** devPtr, size_t size) {
#if defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMallocManaged(devPtr, size));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMallocManaged(devPtr, size));
#endif
}

// wrapper for hip/cuda mem copy
CHAI_HOST inline void  gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind) {
#if defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMemcpy(dst, src, count, kind));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMemcpy(dst, src, count, kind));
#endif
}

#endif //#if defined(CHAI_GPUCC)

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

  CHAISHAREDDLL_API static PointerRecord s_null_record;

  /*!
   * \brief Get the singleton instance.
   *
   * \return Pointer to the ArrayManager instance.
   *
   */
  CHAISHAREDDLL_API
  static ArrayManager* getInstance();

  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  CHAISHAREDDLL_API void setExecutionSpace(ExecutionSpace space);

  /*!
   * \brief Get the current execution space.
   *
   * \return The current execution space.jo
   */
  CHAISHAREDDLL_API ExecutionSpace getExecutionSpace();

  /*!
   * \brief Move data in pointer to the current execution space.
   *
   * \param pointer Pointer to data in any execution space.
   * \return Pointer to data in the current execution space.
   */
  CHAISHAREDDLL_API void* move(void* pointer,
                               PointerRecord* pointer_record,
                               ExecutionSpace = NONE);

  /*!
   * \brief Register a touch of the pointer in the current execution space.
   *
   * \param pointer Raw pointer to register a touch of.
   */
  CHAISHAREDDLL_API void registerTouch(PointerRecord* pointer_record);

  /*!
   * \brief Register a touch of the pointer in the given execution space.
   *
   * The pointer doesn't need to exist in the space being touched.
   *
   * \param pointer Raw pointer to register a touch of.
   * \param space Space to register touch.
   */
  CHAISHAREDDLL_API void registerTouch(PointerRecord* pointer_record, ExecutionSpace space);

  /*!
   * \brief Make a new allocation of the data described by the PointerRecord in
   * the given space.
   *
   * \param pointer_record
   * \param space Space in which to make the allocation.
   */
  CHAISHAREDDLL_API void allocate(PointerRecord* pointer_record, ExecutionSpace space = CPU);

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
  void* reallocate(void* pointer,
                   size_t elems,
                   PointerRecord* record);

  /*!
   * \brief Set the default space for new ManagedArray allocations.
   *
   * ManagedArrays allocated without an explicit ExecutionSpace argument will
   * be allocated in space after this routine is called.
   *
   * \param space New space for default allocations.
   */
  CHAISHAREDDLL_API void setDefaultAllocationSpace(ExecutionSpace space);

  /*!
   * \brief Get the currently set default allocation space.
   *
   * See also setDefaultAllocationSpace.
   *
   * \return Current default space for allocations.
   */
  CHAISHAREDDLL_API ExecutionSpace getDefaultAllocationSpace();

  /*!
   * \brief Free allocation(s) associated with the given PointerRecord.
   *        Default (space == NONE) will free all allocations and delete
   *        the pointer record.
   */
  CHAISHAREDDLL_API void free(PointerRecord* pointer, ExecutionSpace space = NONE);

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
  CHAISHAREDDLL_API size_t getSize(void* pointer);

  CHAISHAREDDLL_API PointerRecord* makeManaged(void* pointer,
                                               size_t size,
                                               ExecutionSpace space,
                                               bool owned);

  /*!
   * \brief Assign a user-defined callback triggered upon memory operations.
   *        This callback applies to a single ManagedArray.
   */
  CHAISHAREDDLL_API void setUserCallback(void* pointer, UserCallback const& f);

  /*!
   * \brief Assign a user-defined callback triggered upon memory operations.
   *        This callback applies to all ManagedArrays.
   */
  CHAISHAREDDLL_API void setGlobalUserCallback(UserCallback const& f);

  /*!
   * \brief Set touched to false in all spaces for the given PointerRecord.
   *
   * \param pointer_record PointerRecord to reset.
   */
  CHAISHAREDDLL_API void resetTouch(PointerRecord* pointer_record);

  /*!
   * \brief Find the PointerRecord corresponding to the raw pointer.
   *
   * \param pointer Raw pointer to find the PointerRecord for.
   *
   * \return PointerRecord containing the raw pointer, or an empty
   *         PointerRecord if none found.
   */
  CHAISHAREDDLL_API PointerRecord* getPointerRecord(void* pointer);

  /*!
   * \brief Create a copy of the given PointerRecord with a new allocation
   *  in the active space.
   *
   * \param record The PointerRecord to copy.
   *
   * \return A copy of the given PointerRecord, must be free'd with delete.
   */
  CHAISHAREDDLL_API PointerRecord* deepCopyRecord(PointerRecord const* record);

  /*!
   * \brief Create a copy of the pointer map.
   *
   * \return A copy of the pointer map. Can be used to find memory leaks.
   */
  CHAISHAREDDLL_API std::unordered_map<void*, const PointerRecord*> getPointerMap() const;

  /*!
   * \brief Get the total number of arrays registered with the array manager.
   *
   * \return The total number of arrays registered with the array manager.
   */
  CHAISHAREDDLL_API size_t getTotalNumArrays() const;

  /*!
   * \brief Get the total amount of memory allocated.
   *
   * \return The total amount of memory allocated.
   */
  CHAISHAREDDLL_API size_t getTotalSize() const;

  /*!
   * \brief Calls callbacks of pointers still in the map with ACTION_LEAKED.
   */
  CHAISHAREDDLL_API void reportLeaks() const;

  /*!
   * \brief Get the allocator ID
   *
   * \return The allocator ID.
   */
  CHAISHAREDDLL_API int getAllocatorId(ExecutionSpace space) const;

  /*!
   * \brief Wraps our resource manager's copy.
   */
  CHAISHAREDDLL_API void copy(void * dst, void * src, size_t size); 
  
  /*!
   * \brief Registering an allocation with the ArrayManager
   *
   * \param record PointerRecord of this allocation.
   * \param space Space in which the pointer was allocated.
   * \param owned Should the allocation be free'd by CHAI?
   */
  CHAISHAREDDLL_API void registerPointer(PointerRecord* record,
                                         ExecutionSpace space,
                                         bool owned = true);

  /*!
   * \brief Deregister a PointerRecord from the ArrayManager.
   *
   * \param record PointerRecord of allocation to deregister.
   * \param deregisterFromUmpire If true, deregister from umpire as well.
   */
  CHAISHAREDDLL_API void deregisterPointer(PointerRecord* record, bool deregisterFromUmpire=false);

  /*!
   * \brief Returns the front of the allocation associated with this pointer, nullptr if allocation not found.
   *
   * \param pointer Pointer to address of that we want the front of the allocation for.
   */
  CHAISHAREDDLL_API void * frontOfAllocation(void * pointer);

  /*!
   * \brief set the allocator for an execution space.
   *
   * \param space Execution space to set the default allocator for.
   * \param allocator The allocator to use for this space. Will be copied into chai.
   */
  void setAllocator(ExecutionSpace space, umpire::Allocator &allocator);

  /*!
   * \brief Get the allocator for an execution space.
   *
   * \param space Execution space of the allocator to get.
   *
   * \return The allocator for the given space.
   */
  umpire::Allocator getAllocator(ExecutionSpace space);
  
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
  [[deprecated("Set the CUDA_LAUNCH_BLOCKING or HIP_LAUNCH_BLOCKING environment variable instead")]]
  void enableDeviceSynchronize() { m_device_synchronize = true; }

  /*!
   * \brief Turn off device synchronization after every kernel.
   */
  [[deprecated("Set the CUDA_LAUNCH_BLOCKING or HIP_LAUNCH_BLOCKING environment variable instead")]]
  void disableDeviceSynchronize() { m_device_synchronize = false; }

  /*!
   * \brief Turn on device synchronization after every kernel.
   */
  bool deviceSynchronize() { return m_device_synchronize; }

  /*!
   * \brief synchronize the device if there hasn't been a synchronize since the last kernel
   */
  CHAISHAREDDLL_API bool syncIfNeeded();

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*!
   * \brief Turn the GPU simulation mode on or off.
   */
  void setGPUSimMode(bool gpuSimMode) { m_gpu_sim_mode = gpuSimMode; }

  /*!
   * \brief Return true if GPU simulation mode is on, false otherwise.
   */
  bool isGPUSimMode() { return m_gpu_sim_mode; }
#endif

  /*!
   * \brief Evicts the data in the given space.
   *
   * \param space Execution space to evict.
   * \param destinationSpace The execution space to move the data to.
   *                            Must not equal space or NONE.
   */
  CHAISHAREDDLL_API void evict(ExecutionSpace space, ExecutionSpace destinationSpace);


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
   * \brief Move data in PointerRecord to the corresponding ExecutionSpace.
   *
   * \param record
   * \param space
   */
  void move(PointerRecord* record, ExecutionSpace space);
  
    /*!
   * \brief Execute a user callback if callbacks are active
   *
   * \param record The pointer record containing the callback
   * \param action The event that occurred
   * \param space The space in which the event occurred
   * \param size The number of bytes in the array associated with this pointer record
   */
  inline void callback(const PointerRecord* record,
                       Action action,
                       ExecutionSpace space) const {
     if (m_callbacks_active) {
        // Callback for this ManagedArray only
        if (record && record->m_user_callback) {
           record->m_user_callback(record, action, space);
        }

        // Callback for all ManagedArrays
        if (m_user_callback) {
           m_user_callback(record, action, space);
        }
     }
  }

  /*!
   * Current execution space.
   */
  static thread_local ExecutionSpace m_current_execution_space;

  /**
   * Default space for new allocations.
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

  /*!
   * \brief The umpire resource manager.
   */
  umpire::ResourceManager& m_resource_manager;

  /*!
   * \brief Used for thread-safe operations.
   */
  mutable std::mutex m_mutex;

  /*!
   * \brief A callback triggered upon memory operations on all ManagedArrays.
   */
  UserCallback m_user_callback;

  /*!
   * \brief Controls whether or not callbacks are called.
   */
  bool m_callbacks_active;

  /*!
   * Whether or not to synchronize on device after every CHAI kernel.
   */
  bool m_device_synchronize = false;

  /*!
   * Whether or not a synchronize has been performed since the launch of the last
   * GPU context
   */
  static thread_local bool m_synced_since_last_kernel;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*!
   * Used by the RAJA plugin to determine whether the execution space should be
   * CPU or GPU.
   */
  bool m_gpu_sim_mode = false;
#endif
};

}  // end of namespace chai

#include "chai/ArrayManager.inl"

#endif  // CHAI_ArrayManager_HPP
