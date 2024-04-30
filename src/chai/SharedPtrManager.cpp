//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/SharedPtrManager.hpp"

#include "chai/ExecutionSpaces.hpp"
#include "chai/config.hpp"

#if defined(CHAI_ENABLE_CUDA)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include "cuda_runtime_api.h"
#endif
#endif

#include "umpire/ResourceManager.hpp"

namespace chai
{
thread_local ExecutionSpace SharedPtrManager::m_current_execution_space;
thread_local bool SharedPtrManager::m_synced_since_last_kernel = false;

msp_pointer_record SharedPtrManager::s_null_record = msp_pointer_record();

SharedPtrManager* SharedPtrManager::getInstance()
{
  static SharedPtrManager s_resource_manager_instance;
  return &s_resource_manager_instance;
}

SharedPtrManager::SharedPtrManager() :
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

void SharedPtrManager::registerPointer(
   msp_pointer_record* record,
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
     msp_pointer_record ** found_pointer_record_addr = found_pointer_record_pair->second;
     if (found_pointer_record_addr != nullptr) {

        msp_pointer_record *foundRecord = *found_pointer_record_addr;
        // if it's actually the same pointer record, then we're OK. If it's a different
        // one, delete the old one.
        if (foundRecord != record) {
           CHAI_LOG(Warning, "SharedPtrManager::registerPointer found a record for " <<
                      pointer << " already there.  Deleting abandoned pointer record.");

           //callback(foundRecord, ACTION_FOUND_ABANDONED, space);

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
         //allocation_record->size = record->m_size;
     }
     // register with umpire if it's not there so that umpire can perform data migrations
     else {
        umpire::util::AllocationRecord new_allocation_record;
        new_allocation_record.ptr = pointer;
        //new_allocation_record.size = record->m_size;
        new_allocation_record.strategy = m_resource_manager.getAllocator(record->m_allocators[space]).getAllocationStrategy();

        m_resource_manager.registerAllocation(pointer, new_allocation_record);
     }
  }
}

void SharedPtrManager::deregisterPointer(msp_pointer_record* record, bool deregisterFromUmpire)
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

void * SharedPtrManager::frontOfAllocation(void * pointer) {
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

void SharedPtrManager::setExecutionSpace(ExecutionSpace space)
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

#if defined(CHAI_THIN_GPU_ALLOCATE)
 if (chai::CPU == space) {
    syncIfNeeded();
 }
#endif

  m_current_execution_space = space;
}

void* SharedPtrManager::move(void* pointer,
                         msp_pointer_record* pointer_record,
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

ExecutionSpace SharedPtrManager::getExecutionSpace()
{
  return m_current_execution_space;
}

void SharedPtrManager::registerTouch(msp_pointer_record* pointer_record)
{
  registerTouch(pointer_record, m_current_execution_space);
}

void SharedPtrManager::registerTouch(msp_pointer_record* pointer_record,
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


void SharedPtrManager::resetTouch(msp_pointer_record* pointer_record)
{
  if (pointer_record && pointer_record!= &s_null_record) {
    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      pointer_record->m_touched[space] = false;
    }
  }
}


/* Not all GPU platform runtimes (notably HIP), will give you asynchronous copies to the device by default, so we leverage
 * umpire's API for asynchronous copies using camp resources in this method, based off of the CHAI destination space
 * */
static void copy(void * dst_pointer, void * src_pointer, umpire::ResourceManager & manager, ExecutionSpace dst_space, ExecutionSpace src_space) {

#ifdef CHAI_ENABLE_CUDA
   camp::resources::Resource device_resource(camp::resources::Cuda::get_default());
#elif defined(CHAI_ENABLE_HIP)
   camp::resources::Resource device_resource(camp::resources::Hip::get_default());
#else
   camp::resources::Resource device_resource(camp::resources::Host::get_default());
#endif

   camp::resources::Resource host_resource(camp::resources::Host::get_default());
   if (dst_space == GPU || src_space == GPU) {
      // Do the copy using the device resource
      manager.copy_poly(dst_pointer, src_pointer, device_resource);
      //manager.copy(dst_pointer, src_pointer);
      //CHAI_GPU_ERROR_CHECK(cudaMemcpyAsync(dst_pointer, src_pointer, 1, cudaMemcpyHostToDevice));
   } else {
      // Do the copy using the host resource
      manager.copy(dst_pointer, src_pointer, host_resource);
   }
   // Ensure device to host copies are synchronous
   if (dst_space == CPU && src_space == GPU) {
      device_resource.wait();
   }
}

void SharedPtrManager::move(msp_pointer_record* record, ExecutionSpace space)
{
  if (space == NONE) {
    return;
  }

  //callback(record, ACTION_CAPTURED, space);

  if (space == record->m_last_space) {
    return;
  }

  ExecutionSpace prev_space = record->m_last_space;

  void* src_pointer = record->m_pointers[prev_space];
  void* dst_pointer = record->m_pointers[space];

  //if (!dst_pointer) {
  //  allocate(record, space);
  //  dst_pointer = record->m_pointers[space];
  //}

  if ( (!record->m_touched[record->m_last_space]) || (! src_pointer )) {
    printf("failed move conditions\n");
    for (int i = chai::CPU; i < NUM_EXECUTION_SPACES; i++) std::cout << i << " : " <<record->m_touched[i] << std::endl;
    std::cout << record->m_last_space << std::endl;
    std::cout << record->m_touched[record->m_last_space] << std::endl;
    std::cout << (src_pointer) << std::endl;
    return;
  } else if (dst_pointer != src_pointer) {
    // Exclude the copy if src and dst are the same (can happen for PINNED memory)
    {
      printf("Performing Copy\n");
      std::cout << "dst_pointer : " << dst_pointer << std::endl;
      std::cout << "src_pointer : " << src_pointer << std::endl;
      std::cout << "space : " << space << std::endl;
      std::cout << "prev_space : " << prev_space << std::endl;
      std::cout << m_resource_manager.findAllocatorForPointer(dst_pointer)->getName() << std::endl;
      std::cout << m_resource_manager.findAllocatorForPointer(src_pointer)->getName() << std::endl;
      chai::copy(dst_pointer, src_pointer, m_resource_manager, space, prev_space);

    }

    //callback(record, ACTION_MOVE, space);
  }

  resetTouch(record);
}

void SharedPtrManager::allocate(
    msp_pointer_record* pointer_record,
           ExecutionSpace space)
{
  //auto size = pointer_record->m_size;
  auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[space]);

  pointer_record->m_pointers[space] = alloc.allocate(1);
  //callback(pointer_record, ACTION_ALLOC, space);
  registerPointer(pointer_record, space);

  CHAI_LOG(Debug, "Allocated array at: " << pointer_record->m_pointers[space]);
}

void SharedPtrManager::free(msp_pointer_record* pointer_record, ExecutionSpace spaceToFree)
{
  if (!pointer_record) return;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (space == spaceToFree || spaceToFree == NONE) {
      if (pointer_record->m_pointers[space]) {
        void* space_ptr = pointer_record->m_pointers[space];
        if (pointer_record->m_owned[space]) {
#if defined(CHAI_ENABLE_UM)
          if (space_ptr == pointer_record->m_pointers[UM]) {
            //callback(pointer_record,
            //         ACTION_FREE,
            //         ExecutionSpace(UM));

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
   //         callback(pointer_record,
   //                  ACTION_FREE,
   //                  ExecutionSpace(space));

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

//size_t SharedPtrManager::getSize(void* ptr)
//{
//  // TODO
//  auto pointer_record = getPointerRecord(ptr);
//  return pointer_record->m_size;
//}

void SharedPtrManager::setDefaultAllocationSpace(ExecutionSpace space)
{
  m_default_allocation_space = space;
}

ExecutionSpace SharedPtrManager::getDefaultAllocationSpace()
{
  return m_default_allocation_space;
}


//void SharedPtrManager::setUserCallback(void* pointer, UserCallback const& f)
//{
//  // TODO ??
//  auto pointer_record = getPointerRecord(pointer);
//  pointer_record->m_user_callback = f;
//}
//
//void SharedPtrManager::setGlobalUserCallback(UserCallback const& f)
//{
//  m_user_callback = f;
//}

msp_pointer_record* SharedPtrManager::getPointerRecord(void* pointer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto record = m_pointer_map.find(pointer);
  return record->second ? *record->second : &s_null_record;
}

msp_pointer_record* SharedPtrManager::makeSharedPtrRecord(void* pointer, void* d_pointer,
                                                          size_t size,
                                                          //ExecutionSpace space,
                                                          bool owned)
{
  if (pointer == nullptr) {
     return &s_null_record ;
  }

  //if (space == NONE) {
  //  space = getDefaultAllocationSpace();
  //}

  m_resource_manager.registerAllocation(
      pointer,
      {pointer, size, m_allocators[chai::CPU]->getAllocationStrategy()});
  std::cout << "m_allocators[chai::CPU] : " << m_allocators[chai::CPU]->getName() << std::endl;

  m_resource_manager.registerAllocation(
      d_pointer,
      {d_pointer, size, m_allocators[chai::GPU]->getAllocationStrategy()});
  std::cout << "m_allocators[chai::GPU] : " << m_allocators[chai::GPU]->getName() << std::endl;

  auto pointer_record = getPointerRecord(pointer);

  if (pointer_record == &s_null_record) {
     if (pointer) {
        pointer_record = new msp_pointer_record();
     } else {
        return pointer_record;
     }
  }
  else {
     CHAI_LOG(Warning, "SharedPtrManager::makeManaged found abandoned pointer record!!!");
     //callback(pointer_record, ACTION_FOUND_ABANDONED, space);
  }

  pointer_record->m_pointers[chai::CPU] = pointer;
  pointer_record->m_owned[chai::CPU] = owned;
  pointer_record->m_pointers[chai::GPU] = d_pointer;
  pointer_record->m_owned[chai::GPU] = owned;
  //pointer_record->m_size = size;
  //pointer_record->m_user_callback = [] (const msp_pointer_record*, Action, ExecutionSpace) {};
  
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_allocators[space] = getAllocatorId(ExecutionSpace(space));
  }

  if (pointer) {
     //registerPointer(pointer_record, chai::CPU, owned);
  }

  return pointer_record;
}

msp_pointer_record* SharedPtrManager::deepCopyRecord(msp_pointer_record const* record)
{
  msp_pointer_record* new_record = new msp_pointer_record{};
  //const size_t size = record->m_size;
  //new_record->m_size = size;
  //new_record->m_user_callback = [] (const msp_pointer_record*, Action, ExecutionSpace) {};

  const ExecutionSpace last_space = record->m_last_space;
  new_record->m_last_space = last_space;
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    new_record->m_allocators[space] = record->m_allocators[space];
  }

  allocate(new_record, last_space);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    new_record->m_owned[space] = true;
    new_record->m_touched[space] = false;
  }

  new_record->m_touched[last_space] = true;

  void* dst_pointer = new_record->m_pointers[last_space];
  void* src_pointer = record->m_pointers[last_space];

  chai::copy(dst_pointer, src_pointer, m_resource_manager, last_space, last_space);

  return new_record;
}

std::unordered_map<void*, const msp_pointer_record*>
SharedPtrManager::getPointerMap() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unordered_map<void*, const msp_pointer_record*> mapCopy;

  for (const auto& entry : m_pointer_map) {
    mapCopy[entry.first] = *entry.second;
  }

  return mapCopy;
}

size_t SharedPtrManager::getTotalNumArrays() const { return m_pointer_map.size(); }

// TODO: Investigate counting memory allocated in each execution space if
// possible
//size_t SharedPtrManager::getTotalSize() const
//{
//  std::lock_guard<std::mutex> lock(m_mutex);
//  size_t total = 0;
//
//  for (const auto& entry : m_pointer_map) {
//    total += (*entry.second)->m_size;
//  }
//
//  return total;
//}

//void SharedPtrManager::reportLeaks() const
//{
//  std::lock_guard<std::mutex> lock(m_mutex);
//  for (const auto& entry : m_pointer_map) {
//    const void* pointer = entry.first;
//    const msp_pointer_record* record = *entry.second;
//
//    for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
//      if (pointer == record->m_pointers[s]) {
//        callback(record, ACTION_LEAKED, ExecutionSpace(s));
//      }
//    }
//  }
//}

int
SharedPtrManager::getAllocatorId(ExecutionSpace space) const
{
  return m_allocators[space]->getId();
}

void SharedPtrManager::evict(ExecutionSpace space, ExecutionSpace destinationSpace) {
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
   std::vector<msp_pointer_record*> pointersToEvict;
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
