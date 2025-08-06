//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_INL
#define CHAI_ManagedArray_INL

#include "ManagedArray.hpp"
#include "ArrayManager.hpp"

namespace chai {

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_size(0),
  m_offset(0),
  m_pointer_record(nullptr),
  m_allocator_id(-1),
  m_is_slice(false)
{
#if !defined(CHAI_DEVICE_COMPILE)
  m_resource_manager = ArrayManager::getInstance();
  m_pointer_record = &ArrayManager::s_null_record;
#endif
}

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray(
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators):
  ManagedArray()
{
  m_pointer_record = new PointerRecord();
  int i = 0;
  for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
    m_pointer_record->m_allocators[s] = m_resource_manager->getAllocatorId(ExecutionSpace(s));
  }

  for (const auto& space : spaces) {
    m_pointer_record->m_allocators[space] = allocators.begin()[i++].getId();
  }

}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems, 
    ExecutionSpace space) :
  ManagedArray()
{
  CHAI_UNUSED_VAR(elems, space);
#if !defined(CHAI_DEVICE_COMPILE)
  this->allocate(elems, space);
#endif
}

template<typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray(
    size_t elems, 
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators,
    ExecutionSpace space):
  ManagedArray(spaces, allocators)
{
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t) :
  ManagedArray()
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST ManagedArray<T>::ManagedArray(PointerRecord* record, ExecutionSpace space):
  m_active_pointer(static_cast<T*>(record->m_pointers[space])),
  m_active_base_pointer(static_cast<T*>(record->m_pointers[space])),
  m_resource_manager(nullptr),
  m_size(record->m_size),
  m_offset(0),
  m_pointer_record(record),
  m_allocator_id(-1),
  m_is_slice(false)
{
   m_resource_manager = ArrayManager::getInstance();
   if (m_pointer_record == nullptr) {
      m_pointer_record = &ArrayManager::s_null_record;
   }
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_active_base_pointer(other.m_active_base_pointer),
  m_resource_manager(other.m_resource_manager),
  m_size(other.m_size),
  m_offset(other.m_offset),
  m_pointer_record(other.m_pointer_record),
  m_allocator_id(-1),
  m_is_slice(other.m_is_slice)
{
#if !defined(CHAI_DEVICE_COMPILE)
  if (m_active_base_pointer || m_size > 0 ) {
     // we only update m_size if we are not null and we have a pointer record
     if (m_pointer_record && !m_is_slice) {
        m_size = m_pointer_record->m_size;
     }
     move(m_resource_manager->getExecutionSpace());
  }
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* array_manager, size_t elems, PointerRecord* pointer_record) :
  m_active_pointer(data), 
  m_active_base_pointer(data),
  m_resource_manager(array_manager),
  m_size(elems*sizeof(T)),
  m_offset(0),
  m_pointer_record(pointer_record),
  m_allocator_id(-1),
  m_is_slice(false)
{
#if !defined(CHAI_DEVICE_COMPILE)
   if (m_resource_manager == nullptr) {
      m_resource_manager = ArrayManager::getInstance();
   }
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   if (m_resource_manager->isGPUSimMode()) {
      return;
   }
#endif
   if (m_pointer_record == &ArrayManager::s_null_record || m_pointer_record==nullptr) {
      m_pointer_record = m_resource_manager->makeManaged((void *) data, m_size,ExecutionSpace(CPU),true);
   }
   registerTouch(CPU);
#endif
}

template <typename T>
CHAI_INLINE
ManagedArray<T> ManagedArray<T>::clone() const
{
  ArrayManager* manager = ArrayManager::getInstance();
  const PointerRecord* record = manager->getPointerRecord(m_active_base_pointer);
  PointerRecord* copy_record = manager->deepCopyRecord(record);
  return ManagedArray(copy_record, copy_record->m_last_space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(
    size_t elems,
    ExecutionSpace space, 
    const UserCallback& cback) 
{
  if(!m_is_slice) {
     if (elems > 0) {
       CHAI_LOG(Debug, "Allocating array of size " << elems << " in space " << space);

       if (m_pointer_record == &ArrayManager::s_null_record) {
         // since we are about to allocate, this will get registered
         m_pointer_record = new PointerRecord();
         for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
           ExecutionSpace allocator_space = space == PINNED ? PINNED : ExecutionSpace(s);
           m_pointer_record->m_allocators[s] = m_resource_manager->getAllocatorId(allocator_space);
         }
       }

       m_pointer_record->m_user_callback = cback;
       m_size = elems*sizeof(T);
       m_pointer_record->m_size = m_size;
       if (space == NONE) {
         space = chai::ArrayManager::getInstance()->getDefaultAllocationSpace();
       }
       m_resource_manager->allocate(m_pointer_record, space);
       m_active_base_pointer = static_cast<T*>(m_pointer_record->m_pointers[space]);
       m_active_pointer = m_active_base_pointer; // Cannot be a slice

       // if T is a CHAICopyable, then it is important to initialize all the
       // ManagedArrays to nullptr at allocation, since it is extremely easy to
       // trigger a moveInnerImpl, which expects inner values to be initialized.
       initInner();
     
#if defined(CHAI_ENABLE_UM)
      if(space == UM) {
        m_pointer_record->m_last_space = UM;
        m_pointer_record->m_pointers[CPU] = m_active_pointer;

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
        m_pointer_record->m_pointers[GPU] = m_active_pointer;
#endif
      }
#endif
#if defined(CHAI_ENABLE_PINNED)
      if (space == PINNED) {
        m_pointer_record->m_last_space = PINNED;
        m_pointer_record->m_pointers[CPU] = m_active_pointer;

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
        m_pointer_record->m_pointers[GPU] = m_active_pointer;
#endif
      }
#endif
       CHAI_LOG(Debug, "m_active_base_ptr allocated at address: " << m_active_base_pointer);
    }
  }
}




template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t elems)
{
  if(!m_is_slice) {
    if (elems > 0) {
      if (m_size == 0 && m_active_base_pointer == nullptr) {
        return allocate(elems, CPU);
      }
      CHAI_LOG(Debug, "Reallocating array of size " << m_size << " bytes with new size" << elems*sizeof(T) << "bytes.");
      if (m_pointer_record == &ArrayManager::s_null_record) {
         m_pointer_record = m_resource_manager->makeManaged((void *)m_active_base_pointer,m_size,CPU,true);
      }
      size_t old_size = m_size;

      m_size = elems*sizeof(T);
      m_active_base_pointer =
        static_cast<T*>(m_resource_manager->reallocate<T>(m_active_base_pointer, elems, m_pointer_record));
      m_active_pointer = m_active_base_pointer; // Cannot be a slice
 
      // if T is a CHAICopyable, then it is important to initialize all the new
      // ManagedArrays to nullptr at allocation, since it is extremely easy to
      // trigger a moveInnerImpl, which expects inner values to be initialized.
      if (initInner(old_size/sizeof(T))) {
        // if we are active on the  GPU, we need to send any newly initialized inner members to the device
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
        if (m_pointer_record->m_last_space == GPU && old_size < m_size) {
          umpire::ResourceManager & umpire_rm = umpire::ResourceManager::getInstance();
          void *src = (void *)(((char *)(m_pointer_record->m_pointers[CPU])) + old_size);
          void *dst = (void *)(((char *)(m_pointer_record->m_pointers[GPU])) + old_size);
          umpire_rm.copy(dst,src,m_size-old_size);
        }
#endif
      }

      CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
    }
    else {
      this->free();
    }
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free(ExecutionSpace space)
{
  if(!m_is_slice) {
    if (m_resource_manager == nullptr) {
       m_resource_manager = ArrayManager::getInstance();
    }
    if (m_pointer_record == &ArrayManager::s_null_record) {
       m_pointer_record = m_resource_manager->makeManaged((void *)m_active_base_pointer,m_size,space,true);
    }
    freeInner();

    m_resource_manager->free(m_pointer_record, space);
    m_active_pointer = nullptr;
    m_active_base_pointer = nullptr;

    // The call to m_resource_manager::free, above, has deallocated m_pointer_record if space == NONE.
    // It's also freed all pointers, so our size and offset should be reset
    if (space == NONE) {
       m_pointer_record = &ArrayManager::s_null_record;
       m_size = 0;
       m_offset = 0;
    }
  } else {
    CHAI_LOG(Debug, "Cannot free a slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reset()
{
  m_resource_manager->resetTouch(m_pointer_record);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE size_t ManagedArray<T>::size() const {
  return m_size/sizeof(T);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::registerTouch(ExecutionSpace space) {
  if (m_active_pointer && (m_pointer_record == nullptr || m_pointer_record == &ArrayManager::s_null_record)) {
     CHAI_LOG(Warning,"registerTouch called on ManagedArray with nullptr pointer record.");
     m_pointer_record = m_resource_manager->makeManaged((void *)m_active_base_pointer,m_size,space,true);
  }
  m_resource_manager->registerTouch(m_pointer_record, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
typename ManagedArray<T>::T_non_const ManagedArray<T>::pick(size_t i) const { 
  #if !defined(CHAI_DEVICE_COMPILE)
    #if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      if (m_resource_manager->isGPUSimMode()) {
        return (T_non_const)(m_active_pointer[i]);
      }
    #endif
    #if defined(CHAI_ENABLE_UM)
      if(m_pointer_record->m_pointers[UM] == m_active_base_pointer) {
        synchronize();
        return (T_non_const)(m_active_pointer[i]);
      }
    #endif
    ExecutionSpace last_space = m_pointer_record->m_last_space;
    if (last_space == NONE || last_space == CPU) {
       return ((T*)m_pointer_record->m_pointers[CPU])[i+m_offset];
    }
    else {
       T * addr = (T*)m_pointer_record->m_pointers[last_space];
       return m_resource_manager->pick(addr, i+m_offset);
    }
  #else
    return (T_non_const)(m_active_pointer[i]); 
  #endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T val) const { 
  #if !defined(CHAI_DEVICE_COMPILE)
    #if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      if (m_resource_manager->isGPUSimMode()) {
        m_active_pointer[i] = val;
        return;
      }
    #endif
    #if defined(CHAI_ENABLE_UM)
      if(m_pointer_record->m_pointers[UM] == m_active_pointer) {
        synchronize();
        m_active_pointer[i] = val;
        return;
      }
    #endif

    if (m_pointer_record->m_last_space == NONE) {
       // Use the first non-null pointer if managed array was not used yet
       for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
          if (m_pointer_record->m_pointers[s] != nullptr) {
             m_pointer_record->m_last_space = static_cast<ExecutionSpace>(s);
             break;
          }
       }
    }

    m_pointer_record->m_touched[m_pointer_record->m_last_space] = true;
    m_resource_manager->set(static_cast<T*>((void*)((char*)m_pointer_record->m_pointers[m_pointer_record->m_last_space]+sizeof(T)*m_offset)), i, val);
  #else
    m_active_pointer[i] = val; 
  #endif // !defined(CHAI_DEVICE_COMPILE)
}

template <typename T>
CHAI_INLINE
CHAI_HOST
void ManagedArray<T>::move(ExecutionSpace space, bool registerTouch) const
{
  if (m_pointer_record != &ArrayManager::s_null_record) {
     ExecutionSpace prev_space = m_pointer_record->m_last_space;
     if (prev_space == CPU || prev_space == NONE) {
        /// Move nested ManagedArrays first, so they are working with a valid m_active_pointer for the host,
        // and so the meta data associated with them are updated before we move the other array down.
        moveInnerImpl();
     }
     CHAI_LOG(Debug, "Moving " << m_active_pointer);
     m_active_base_pointer = static_cast<T*>(m_resource_manager->move((void *)m_active_base_pointer, m_pointer_record, space));
     m_active_pointer = m_active_base_pointer + m_offset;

     CHAI_LOG(Debug, "Moved to " << m_active_pointer);
#if defined(CHAI_ENABLE_UM)
    if (m_pointer_record->m_last_space == UM) {
       // just because we were allocated in UM doesn't mean our CHAICopyable array values were
       moveInnerImpl();
    } else
#endif
#if defined(CHAI_ENABLE_PINNED)
    if (m_pointer_record->m_last_space == PINNED) {
       // just because we were allocated in PINNED doesn't mean our CHAICopyable array values were
       moveInnerImpl();
    } else 
#endif
     if (registerTouch) {
       CHAI_LOG(Debug, "T is non-const, registering touch of pointer" << m_active_pointer);
       m_resource_manager->registerTouch(m_pointer_record, space);
     }
     if (space != GPU && prev_space == GPU) {
        /// Move nested ManagedArrays after the move, so they are working with a valid m_active_pointer for the host,
        // and so the meta data associated with them are updated with live GPU data
        moveInnerImpl();
     }
   }
}

template<typename T>
template<typename Idx>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const Idx i) const {
  return m_active_pointer[i];
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
T* ManagedArray<T>::data() const {
#if !defined(CHAI_DEVICE_COMPILE)
  if (m_active_pointer) {
     if (m_pointer_record == nullptr || m_pointer_record == &ArrayManager::s_null_record) {
        CHAI_LOG(Warning, "nullptr pointer_record associated with non-nullptr active_pointer")
     }

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
     if (m_resource_manager->isGPUSimMode()) {
        return m_active_pointer;
     }
#endif
     move(CPU);
  }

  if (m_size == 0 && !m_is_slice) {
     return nullptr;
  }

  return m_active_pointer;
#else
  return m_active_pointer;
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
const T* ManagedArray<T>::cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
  if (m_active_pointer) {
     if (m_pointer_record == nullptr || m_pointer_record == &ArrayManager::s_null_record) {
        CHAI_LOG(Warning, "nullptr pointer_record associated with non-nullptr active_pointer")
     }

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
     if (m_resource_manager->isGPUSimMode()) {
        return m_active_pointer;
     }
#endif
     move(CPU, false);
  }

  if (m_size == 0 && !m_is_slice) {
     return nullptr;
  }

  return m_active_pointer;
#else
  return m_active_pointer;
#endif
}

template<typename T>
T* ManagedArray<T>::data(ExecutionSpace space, bool do_move) const {
   if (m_pointer_record == nullptr || m_pointer_record == &ArrayManager::s_null_record) {
      return nullptr;
   }

   if (m_size == 0 && !m_is_slice) {
      return nullptr;
   }

   if (do_move) {
      ExecutionSpace oldContext = m_resource_manager->getExecutionSpace();
      m_resource_manager->setExecutionSpace(space);
      move(space);
      m_resource_manager->setExecutionSpace(oldContext);
   }

   int offset = m_is_slice ? m_offset : 0 ;
   return ((T*) m_pointer_record->m_pointers[space]) + offset;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE T* ManagedArray<T>::begin() const {
   return data();
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE T* ManagedArray<T>::end() const {
   return data() + size();
}

//template<typename T>
//ManagedArray<T>::operator ManagedArray<
//  typename std::conditional<!std::is_const<T>::value, 
//                            const T, 
//                            InvalidConstCast>::type> ()const
template< typename T>
template< typename U>
ManagedArray<T>::operator 
typename std::enable_if< !std::is_const<U>::value , 
                         ManagedArray<const U> >::type () const

{
  return *reinterpret_cast<ManagedArray<const T> *>(const_cast<ManagedArray<T> *>(this));
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>&
ManagedArray<T>::operator= (ManagedArray && other) {
  *this = other;
  other = nullptr;
  return *this;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>&
ManagedArray<T>::operator= (std::nullptr_t) {
  m_active_pointer = nullptr;
  m_active_base_pointer = nullptr;
  m_size = 0;
  m_offset = 0;
  #if !defined(CHAI_DEVICE_COMPILE)
  m_pointer_record = &ArrayManager::s_null_record;
  m_resource_manager = ArrayManager::getInstance();
  #else
  m_pointer_record = nullptr;
  m_resource_manager = nullptr;
  #endif
  m_is_slice = false;
  return *this;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator== (const ManagedArray<T>& rhs) const
{
  return (m_active_pointer ==  rhs.m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator!= (const ManagedArray<T>& rhs) const
{
  return (m_active_pointer !=  rhs.m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator== (std::nullptr_t from) const {
   return m_active_pointer == from || m_size == 0;
}
template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator!= (std::nullptr_t from) const {
   return m_active_pointer != from && m_size > 0;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>::operator bool () const {
   return m_size > 0;
}

template<typename T>
template<bool B, typename std::enable_if<B, int>::type>
CHAI_INLINE
CHAI_HOST
void
ManagedArray<T>::moveInnerImpl() const
{
  int len = m_pointer_record->m_size / sizeof(T);
  T * host_ptr = (T *) m_pointer_record->m_pointers[CPU]; 
  for (int i = 0; i < len; ++i)
  {
    // trigger the copy constructor
    T inner = T(host_ptr[i]);
    // ensure the inner type gets the state of the result of the copy
    host_ptr[i].shallowCopy(inner);
  }
}

template<typename T>
template<bool B, typename std::enable_if<!B, int>::type>
CHAI_INLINE
CHAI_HOST
void
ManagedArray<T>::moveInnerImpl() const
{
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
