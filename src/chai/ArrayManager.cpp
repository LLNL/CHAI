//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/ArrayManager.hpp"

#include "chai/config.hpp"

#if defined(CHAI_ENABLE_CUDA)
#include "cuda_runtime_api.h"
#endif

#include "umpire/ResourceManager.hpp"

namespace chai
{
thread_local ExecutionSpace ArrayManager::m_current_execution_space;
thread_local bool ArrayManager::m_synced_since_last_kernel = false;

PointerRecord ArrayManager::s_null_record = PointerRecord();

ArrayManager* ArrayManager::getInstance()
{
  static ArrayManager s_resource_manager_instance;
  return &s_resource_manager_instance;
}

ArrayManager::ArrayManager() :
  m_pointer_map{},
  m_allocators{},
  m_resource_manager{umpire::ResourceManager::getInstance()},
  m_callbacks_active{true}
{
  m_pointer_map.clear();
  m_current_execution_space = NONE;
  m_default_allocation_space = CPU;

  m_allocators[CPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  m_allocators[GPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));
#else
  m_allocators[GPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("DEVICE"));
#endif
#endif

#if defined(CHAI_ENABLE_UM)
  m_allocators[UM] =
      new umpire::Allocator(m_resource_manager.getAllocator("UM"));
#endif

#if defined(CHAI_ENABLE_PINNED)
#if (defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)) && !defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
    m_allocators[PINNED] =
             new umpire::Allocator(m_resource_manager.getAllocator("PINNED"));
#else
  m_allocators[PINNED] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));
#endif
#endif
}

void ArrayManager::registerPointer(
   PointerRecord* record,
   ExecutionSpace space,
   bool owned)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto pointer = record->m_pointers[space];

  // if we are registering a new pointer record for a pointer where there is already
  // a pointer record, we assume the old record was somehow abandoned by the host
  // application and trigger an ACTION_FOUND_ABANDONED callback
  auto found_pointer_record_pair = m_pointer_map.find(pointer);
  if (found_pointer_record_pair != m_pointer_map.end()) {
     PointerRecord ** found_pointer_record_addr = found_pointer_record_pair->second;
     if (found_pointer_record_addr != nullptr) {

        PointerRecord *foundRecord = *found_pointer_record_addr;
        // if it's actually the same pointer record, then we're OK. If it's a different
        // one, delete the old one.
        if (foundRecord != record) {
           CHAI_LOG(Warning, "ArrayManager::registerPointer found a record for " <<
                      pointer << " already there.  Deleting abandoned pointer record.");

           callback(foundRecord, ACTION_FOUND_ABANDONED, space);

           for (int fspace = CPU; fspace < NUM_EXECUTION_SPACES; ++fspace) {
              foundRecord->m_pointers[fspace] = nullptr;
           }

           delete foundRecord;
        }
     }
  }

  CHAI_LOG(Debug, "Registering " << pointer << " in space " << space);

  m_pointer_map.insert(pointer, record);

  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    if (!record->m_pointers[i]) record->m_owned[i] = true;
  }
  record->m_owned[space] = owned;

  if (pointer) {
     // if umpire already knows about this pointer, we want to make sure its records and ours
     // are consistent
     if (m_resource_manager.hasAllocator(pointer)) {
         umpire::util::AllocationRecord *allocation_record = const_cast<umpire::util::AllocationRecord *>(m_resource_manager.findAllocationRecord(pointer));
         allocation_record->size = record->m_size;
     }
     // register with umpire if it's not there so that umpire can perform data migrations
     else {
        umpire::util::AllocationRecord new_allocation_record;
        new_allocation_record.ptr = pointer;
        new_allocation_record.size = record->m_size;
        new_allocation_record.strategy = m_resource_manager.getAllocator(record->m_allocators[space]).getAllocationStrategy();

        m_resource_manager.registerAllocation(pointer, new_allocation_record);
     }
  }
}

void ArrayManager::deregisterPointer(PointerRecord* record, bool deregisterFromUmpire)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    void * pointer = record->m_pointers[i];
    if (pointer) {
       if (deregisterFromUmpire) {
          m_resource_manager.deregisterAllocation(pointer);
       }
       CHAI_LOG(Debug, "De-registering " << pointer);
       m_pointer_map.erase(pointer);
    }
  }
  if (record != &s_null_record) {
     delete record;
  }
}

void * ArrayManager::frontOfAllocation(void * pointer) {
  if (pointer) {
    if (m_resource_manager.hasAllocator(pointer)) {
       auto allocation_record = m_resource_manager.findAllocationRecord(pointer);
       if (allocation_record) {
         return allocation_record->ptr;
       }
    }
  }
  return nullptr;
}

void ArrayManager::setExecutionSpace(ExecutionSpace space)
{
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   if (isGPUSimMode()) {
      space = chai::GPU;
   }
#endif

  CHAI_LOG(Debug, "Setting execution space to " << space);

  if (chai::GPU == space) {
    m_synced_since_last_kernel = false;
  }

  m_current_execution_space = space;
}

void* ArrayManager::move(void* pointer,
                         PointerRecord* pointer_record,
                         ExecutionSpace space)
{
  // Check for default arg (NONE)
  if (space == NONE) {
    space = m_current_execution_space;
  }

  if (space == NONE) {
    return pointer;
  }

  move(pointer_record, space);

  return pointer_record->m_pointers[space];
}

ExecutionSpace ArrayManager::getExecutionSpace()
{
  return m_current_execution_space;
}

void ArrayManager::registerTouch(PointerRecord* pointer_record)
{
  registerTouch(pointer_record, m_current_execution_space);
}

void ArrayManager::registerTouch(PointerRecord* pointer_record,
                                 ExecutionSpace space)
{
  if (pointer_record && pointer_record != &s_null_record) {

     if (space != NONE) {
       CHAI_LOG(Debug, pointer_record->m_pointers[space] << " touched in space " << space);
       pointer_record->m_touched[space] = true;
       pointer_record->m_last_space = space;
     }
  }
}


void ArrayManager::resetTouch(PointerRecord* pointer_record)
{
  if (pointer_record && pointer_record!= &s_null_record) {
    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      pointer_record->m_touched[space] = false;
    }
  }
}

void ArrayManager::move(PointerRecord* record, ExecutionSpace space)
{
  if (space == NONE) {
    return;
  }

  callback(record, ACTION_CAPTURED, space);

  if (space == record->m_last_space) {
    return;
  }

#if defined(CHAI_ENABLE_UM)
  if (record->m_last_space == UM) {
    return;
  }
#endif

#if defined(CHAI_ENABLE_PINNED)
  if (record->m_last_space == PINNED) {
    if (space == CPU) {
      syncIfNeeded();
    }
    return;
  }
#endif

  void* src_pointer = record->m_pointers[record->m_last_space];
  void* dst_pointer = record->m_pointers[space];

  if (!dst_pointer) {
    allocate(record, space);
    dst_pointer = record->m_pointers[space];
  }


  if ( (!record->m_touched[record->m_last_space]) || (! src_pointer )) {
    return;
  } else if (dst_pointer != src_pointer) {
    // Exclude the copy if src and dst are the same (can happen for PINNED memory)
    {
      m_resource_manager.copy(dst_pointer, src_pointer);
    }

    callback(record, ACTION_MOVE, space);
  }

  resetTouch(record);
}

void ArrayManager::allocate(
    PointerRecord* pointer_record,
           ExecutionSpace space)
{
  auto size = pointer_record->m_size;
  auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[space]);

  pointer_record->m_pointers[space] = alloc.allocate(size);
  callback(pointer_record, ACTION_ALLOC, space);

  registerPointer(pointer_record, space);

  CHAI_LOG(Debug, "Allocated array at: " << pointer_record->m_pointers[space]);
}

void ArrayManager::free(PointerRecord* pointer_record, ExecutionSpace spaceToFree)
{
  if (!pointer_record) return;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (space == spaceToFree || spaceToFree == NONE) {
      if (pointer_record->m_pointers[space]) {
        void* space_ptr = pointer_record->m_pointers[space];
        if (pointer_record->m_owned[space]) {
#if defined(CHAI_ENABLE_UM)
          if (space_ptr == pointer_record->m_pointers[UM]) {
            callback(pointer_record,
                     ACTION_FREE,
                     ExecutionSpace(UM));

            auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[UM]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
#if defined(CHAI_ENABLE_PINNED)
          if (space_ptr == pointer_record->m_pointers[PINNED]) {
            callback(pointer_record,
                     ACTION_FREE,
                     ExecutionSpace(PINNED));

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[PINNED]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
          {
            callback(pointer_record,
                     ACTION_FREE,
                     ExecutionSpace(space));

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[space]);
            alloc.deallocate(space_ptr);

            pointer_record->m_pointers[space] = nullptr;
          }
        }
        else
        {
          m_resource_manager.deregisterAllocation(space_ptr);
        }
        {
          CHAI_LOG(Debug, "DeRegistering " << space_ptr);
          std::lock_guard<std::mutex> lock(m_mutex);
          m_pointer_map.erase(space_ptr);
        }
      }
    }
  }
  
  if (pointer_record != &s_null_record && spaceToFree == NONE) {
    delete pointer_record;
  }
}

size_t ArrayManager::getSize(void* ptr)
{
  // TODO
  auto pointer_record = getPointerRecord(ptr);
  return pointer_record->m_size;
}

void ArrayManager::setDefaultAllocationSpace(ExecutionSpace space)
{
  m_default_allocation_space = space;
}

ExecutionSpace ArrayManager::getDefaultAllocationSpace()
{
  return m_default_allocation_space;
}


void ArrayManager::setUserCallback(void* pointer, UserCallback const& f)
{
  // TODO ??
  auto pointer_record = getPointerRecord(pointer);
  pointer_record->m_user_callback = f;
}

void ArrayManager::setGlobalUserCallback(UserCallback const& f)
{
  m_user_callback = f;
}

PointerRecord* ArrayManager::getPointerRecord(void* pointer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto record = m_pointer_map.find(pointer);
  return record->second ? *record->second : &s_null_record;
}

PointerRecord* ArrayManager::makeManaged(void* pointer,
                                         size_t size,
                                         ExecutionSpace space,
                                         bool owned)
{
  if (pointer == nullptr) {
     return &s_null_record ;
  }

  if (space == NONE) {
    space = getDefaultAllocationSpace();
  }

  m_resource_manager.registerAllocation(
      pointer,
      {pointer, size, m_allocators[space]->getAllocationStrategy()});

  auto pointer_record = getPointerRecord(pointer);

  if (pointer_record == &s_null_record) {
     if (pointer) {
        pointer_record = new PointerRecord();
     } else {
        return pointer_record;
     }
  }
  else {
     CHAI_LOG(Warning, "ArrayManager::makeManaged found abandoned pointer record!!!");
     callback(pointer_record, ACTION_FOUND_ABANDONED, space);
  }

  pointer_record->m_pointers[space] = pointer;
  pointer_record->m_owned[space] = owned;
  pointer_record->m_size = size;
  pointer_record->m_user_callback = [] (const PointerRecord*, Action, ExecutionSpace) {};
  
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_allocators[space] = getAllocatorId(ExecutionSpace(space));
  }

  if (pointer && size > 0) {
     registerPointer(pointer_record, space, owned);
  }

  return pointer_record;
}

PointerRecord* ArrayManager::deepCopyRecord(PointerRecord const* record)
{
  PointerRecord* copy = new PointerRecord{};
  const size_t size = record->m_size;
  copy->m_size = size;
  copy->m_user_callback = [] (const PointerRecord*, Action, ExecutionSpace) {};

  const ExecutionSpace last_space = record->m_last_space;
  copy->m_last_space = last_space;
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    copy->m_allocators[space] = record->m_allocators[space];
  }

  allocate(copy, last_space);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    copy->m_owned[space] = true;
    copy->m_touched[space] = false;
  }

  copy->m_touched[last_space] = true;

  void* dst_pointer = copy->m_pointers[last_space];
  void* src_pointer = record->m_pointers[last_space];

  m_resource_manager.copy(dst_pointer, src_pointer);

  return copy;
}

std::unordered_map<void*, const PointerRecord*>
ArrayManager::getPointerMap() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unordered_map<void*, const PointerRecord*> mapCopy;

  for (const auto& entry : m_pointer_map) {
    mapCopy[entry.first] = *entry.second;
  }

  return mapCopy;
}

size_t ArrayManager::getTotalNumArrays() const { return m_pointer_map.size(); }

// TODO: Investigate counting memory allocated in each execution space if
// possible
size_t ArrayManager::getTotalSize() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  size_t total = 0;

  for (const auto& entry : m_pointer_map) {
    total += (*entry.second)->m_size;
  }

  return total;
}

void ArrayManager::reportLeaks() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  for (const auto& entry : m_pointer_map) {
    const void* pointer = entry.first;
    const PointerRecord* record = *entry.second;

    for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
      if (pointer == record->m_pointers[s]) {
        callback(record, ACTION_LEAKED, ExecutionSpace(s));
      }
    }
  }
}

int
ArrayManager::getAllocatorId(ExecutionSpace space) const
{
  return m_allocators[space]->getId();
}

void ArrayManager::evict(ExecutionSpace space, ExecutionSpace destinationSpace) {
   // Check arguments
   if (space == NONE) {
      // Nothing to be done
      return;
   }

   if (destinationSpace == NONE) {
      // If the destination space is NONE, evicting invalidates all data and
      // leaves us in a bad state (if the last touch was in the eviction space).
      CHAI_LOG(Warning, "evict does nothing with destinationSpace == NONE!");
      return;
   }

   if (space == destinationSpace) {
      // It doesn't make sense to evict to the same space, so do nothing
      CHAI_LOG(Warning, "evict does nothing with space == destinationSpace!");
      return;
   }

   // Now move and evict
   std::vector<PointerRecord*> pointersToEvict;
   {
      std::lock_guard<std::mutex> lock(m_mutex);
      for (const auto& entry : m_pointer_map) {
         // Get the pointer record
         auto record = *entry.second;

         // Move the data and register the touches
         move(record, destinationSpace);
         registerTouch(record, destinationSpace);

         // If the destinationSpace is ever allowed to be NONE, then we will need to
         // update the touch in the eviction space and make sure the last space is not
         // the eviction space.

         // Mark record for eviction later in this routine
         pointersToEvict.push_back(record);
      }
   }

   // This must be done in a second pass because free erases from m_pointer_map,
   // which would invalidate the iterator in the above loop
   for (const auto& entry : pointersToEvict) {
      free(entry, space);
   }
}


}  // end of namespace chai
