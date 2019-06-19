// ---------------------------------------------------------------------
// Copyright (c) 2017, Lawrence Livermore National Security, LLC. All
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
#include "chai/ArrayManager.hpp"

#include "chai/config.hpp"

#include "umpire/ResourceManager.hpp"

namespace chai
{

PointerRecord ArrayManager::s_null_record = PointerRecord();

ArrayManager* ArrayManager::getInstance()
{
  static ArrayManager s_resource_manager_instance;
  return &s_resource_manager_instance;
}

ArrayManager::ArrayManager() :
  m_pointer_map{},
  m_allocators{},
  m_resource_manager{umpire::ResourceManager::getInstance()}
{
  m_pointer_map.clear();
  m_current_execution_space = NONE;
  m_default_allocation_space = CPU;

  m_allocators[CPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));
#if defined(CHAI_ENABLE_CUDA)
  m_allocators[GPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("DEVICE"));
#endif
#if defined(CHAI_ENABLE_UM)
  m_allocators[UM] =
      new umpire::Allocator(m_resource_manager.getAllocator("UM"));
#endif
}

void ArrayManager::registerPointer(
   PointerRecord* record,
   ExecutionSpace space,
   bool owned)
{
  CHAI_LOG("ArrayManager", "Registering " << pointer << " in space " << space);

  auto pointer = record->m_pointers[space];

  m_pointer_map.insert(pointer, record);
  //record->m_last_space = space;

  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    record->m_owned[i] = true;
  }
  record->m_owned[space] = owned;
}


void ArrayManager::deregisterPointer(PointerRecord* record)
{
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    if (record->m_pointers[i]) m_pointer_map.erase(record->m_pointers[i]);
  }

  delete record;
}

void ArrayManager::setExecutionSpace(ExecutionSpace space)
{
  CHAI_LOG("ArrayManager", "Setting execution space to " << space);

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
  CHAI_LOG("ArrayManager",
           pointer << " touched in space " << space);

  if (space == NONE) return;

  pointer_record->m_touched[space] = true;
  pointer_record->m_last_space = space;
}


void ArrayManager::resetTouch(PointerRecord* pointer_record)
{
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_touched[space] = false;
  }
}

void ArrayManager::move(PointerRecord* record, ExecutionSpace space)
{
  if (space == NONE) {
    return;
  }

#if defined(CHAI_ENABLE_UM)
  if (record->m_last_space == UM) {
    return;
  }
#endif

  if (space == record->m_last_space) {
    return;
  }


  void* src_pointer = record->m_pointers[record->m_last_space];
  void* dst_pointer = record->m_pointers[space];

  if (!dst_pointer) {
    allocate(record, space);
    dst_pointer = record->m_pointers[space];
  }

  if (!record->m_touched[record->m_last_space]) {
    return;
  } else {
    record->m_user_callback(ACTION_MOVE, space, record->m_size);
    m_resource_manager.copy(dst_pointer, src_pointer);
  }

  resetTouch(record);
}

void ArrayManager::allocate(
    PointerRecord* pointer_record,
           ExecutionSpace space)
{
  auto size = pointer_record->m_size;
  auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[space]);

  pointer_record->m_user_callback(ACTION_ALLOC, space, size);

  pointer_record->m_pointers[space] =  alloc.allocate(size);
  registerPointer(pointer_record, space);

  CHAI_LOG("ArrayManager", "Allocated array at: " << ret);
}

void ArrayManager::free(PointerRecord* pointer_record)
{
  if (pointer_record == nullptr) return;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (pointer_record->m_pointers[space]) {
      if (pointer_record->m_owned[space]) {
        void* space_ptr = pointer_record->m_pointers[space];
#if defined(CHAI_ENABLE_UM)
        if (space_ptr == pointer_record->m_pointers[UM]) {
          pointer_record->m_user_callback(ACTION_FREE,
                                          ExecutionSpace(UM),
                                          pointer_record->m_size);
          m_pointer_map.erase(space_ptr);

          auto alloc = m_resource_manager.getAllocator(
              pointer_record->m_allocators[space]);
          alloc.deallocate(space_ptr);

          for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
            if (space_ptr == pointer_record->m_pointers[space_t])
              pointer_record->m_pointers[space_t] = nullptr;
          }
        } else {
#endif
          pointer_record->m_user_callback(ACTION_FREE,
                                          ExecutionSpace(space),
                                          pointer_record->m_size);
          m_pointer_map.erase(space_ptr);

          auto alloc = m_resource_manager.getAllocator(
              pointer_record->m_allocators[space]);
          alloc.deallocate(space_ptr);

          pointer_record->m_pointers[space] = nullptr;
#if defined(CHAI_ENABLE_UM)
        }
#endif
      }
      else
      {
        m_resource_manager.deregisterAllocation(pointer_record->m_pointers[space]);
      }
    }
  }

  delete pointer_record;
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

PointerRecord* ArrayManager::getPointerRecord(void* pointer)
{
  auto record = m_pointer_map.find(pointer);
  return record->second ? *record->second : &s_null_record;
}

PointerRecord* ArrayManager::makeManaged(void* pointer,
                                         size_t size,
                                         ExecutionSpace space,
                                         bool owned)
{
  m_resource_manager.registerAllocation(
      pointer,
      {pointer, size, m_allocators[space]->getAllocationStrategy()});

  auto pointer_record = new PointerRecord{};

  pointer_record->m_pointers[space] = pointer;
  pointer_record->m_owned[space] = owned;
  pointer_record->m_size = size;
  pointer_record->m_user_callback = [](Action, ExecutionSpace, size_t) {};

  registerPointer(pointer_record, space, owned);

  // TODO Is this a problem?
  // for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
  //   // If pointer is already active on some execution space, return that
  //   pointer if(pointer_record->m_touched[i] == true)
  //     return pointer_record->m_pointers[i];
  // }

  return pointer_record;
}

PointerRecord* ArrayManager::deepCopyRecord(PointerRecord const* record)
{
  PointerRecord* copy = new PointerRecord();
  const size_t size = record->m_size;
  copy->m_size = size;
  copy->m_user_callback = [](Action, ExecutionSpace, size_t) {};

  const ExecutionSpace last_space = record->m_last_space;

  copy->m_last_space = last_space;
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
  std::unordered_map<void*, const PointerRecord*> mapCopy;

  for (auto entry : m_pointer_map) {
    mapCopy[entry.first] = *entry.second;
  }

  return mapCopy;
}

size_t ArrayManager::getTotalNumArrays() const { return m_pointer_map.size(); }

// TODO: Investigate counting memory allocated in each execution space if
// possible
size_t ArrayManager::getTotalSize() const
{
  size_t total = 0;

  for (auto entry : m_pointer_map) {
    total += (*entry.second)->m_size;
  }

  return total;
}

int
ArrayManager::getAllocatorId(ExecutionSpace space) const
{
  return m_allocators[space]->getId();

}

}  // end of namespace chai
