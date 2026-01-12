//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_thin_INL
#define CHAI_ManagedArray_thin_INL

#include "ManagedArray.hpp"

#if defined(CHAI_ENABLE_UM)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include <cuda_runtime_api.h>
#endif
#endif

namespace chai {

template <typename T>
CHAI_INLINE ManagedArray<T>::ManagedArray(
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators)
    : ManagedArray()
{
  int i = 0;
  auto allocators_begin = allocators.begin();
  auto default_space = chai::ArrayManager::getInstance()->getDefaultAllocationSpace();
  for (const auto& space : spaces) {
    if (space == default_space) {
       m_allocator_id = allocators_begin[i].getId();
    }
    ++i;
  }
}

template <typename T>
CHAI_INLINE
ManagedArray<T>::ManagedArray(
    size_t elems,
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators,
    ExecutionSpace space) :
  ManagedArray(spaces, allocators)
{
  m_size = elems*sizeof(T);
  int i = 0;
  auto allocators_begin = allocators.begin();
  for (const auto& s : spaces) {
     if (s == space) {
       m_allocator_id = allocators_begin[i].getId();
     }
     ++i;
  }
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray() = default;

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(size_t elems, ExecutionSpace space) :
  m_size(elems*sizeof(T))
{
#ifndef CHAI_DEVICE_COMPILE
  this->allocate(elems, space);
#endif
}


template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t)
    : m_active_pointer(nullptr),
      m_active_base_pointer(nullptr),
      m_resource_manager(nullptr),
      m_size(0),
      m_offset(0),
      m_pointer_record(nullptr),
      m_allocator_id(-1),
      m_is_slice(false)
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
  m_pointer_record(nullptr),
  m_allocator_id(-1),
  m_is_slice(!record->m_owned[space])
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other) = default;

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data,
                                               ArrayManager* array_manager,
                                               size_t elems,
                                               PointerRecord* pointer_record) :
  m_active_pointer(data), 
  m_active_base_pointer(data),
  m_resource_manager(array_manager),
  m_size(elems*sizeof(T)),
  m_pointer_record(pointer_record),
  m_allocator_id(-1)
{
   if (pointer_record) {
      m_allocator_id = pointer_record->m_allocators[pointer_record->m_last_space];
   }
}

template <typename T>
CHAI_INLINE
ManagedArray<T> ManagedArray<T>::clone() const
{
  ManagedArray<T> result;
  result.m_allocator_id = m_allocator_id;
  result.allocate(size());
  auto arrayManager = ArrayManager::getInstance();
  arrayManager->syncIfNeeded();
  arrayManager->copy(result.m_active_pointer, m_active_pointer, m_size);
  result.registerTouch(chai::GPU);
  return result;
}

template <typename T>
CHAI_HOST_DEVICE T* ManagedArray<T>::data() const
{
#if !defined(CHAI_DEVICE_COMPILE) && defined(CHAI_THIN_GPU_ALLOCATE)
   ArrayManager::getInstance()->syncIfNeeded();
#endif
   return m_active_pointer;
}

template <typename T>
CHAI_HOST_DEVICE const T* ManagedArray<T>::cdata() const
{
#if !defined(CHAI_DEVICE_COMPILE) && defined(CHAI_THIN_GPU_ALLOCATE)
   ArrayManager::getInstance()->syncIfNeeded();
#endif
   return m_active_pointer;
}

template <typename T>
T* ManagedArray<T>::data(ExecutionSpace space, bool do_move) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
  if (do_move && space != chai::GPU) {
      ArrayManager::getInstance()->syncIfNeeded();
  }
#endif
  return m_active_pointer;
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

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(size_t elems,
                                         ExecutionSpace space,
                                         const UserCallback&) {
  if (!m_is_slice) {
     if (m_allocator_id == -1) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
       if (space == NONE) {
          space = chai::ArrayManager::getInstance()->getDefaultAllocationSpace();
       }

#elif defined(CHAI_ENABLE_UM)
       space = chai::UM;
#else
       space = chai::CPU;
#endif

       m_allocator_id = chai::ArrayManager::getInstance()->getAllocatorId(space);
     }

     if (elems > 0) {
       CHAI_LOG(Debug, "Allocating array of size " << elems
                                                   << " in space "
                                                   << space);

       auto allocator = chai::ArrayManager::getInstance()->getAllocator(m_allocator_id);
       m_size = elems*sizeof(T);
       m_active_pointer = (T*) allocator.allocate(m_size);

       CHAI_LOG(Debug, "m_active_ptr allocated at address: " << m_active_pointer);
     }
     else {
        m_active_pointer = nullptr;
        m_size = 0;
     }
  }
  else {
    CHAI_LOG(Debug, "Attempted to allocate slice!");
  }

  m_active_base_pointer = m_active_pointer;
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t new_elems)
{
  if (!m_is_slice) {
    // reallocating a nullptr is equivalent to an allocate
    if (m_size == 0 && m_active_base_pointer == nullptr) {
      return allocate(new_elems);
    }

    if (m_size != new_elems*sizeof(T)) {
       CHAI_LOG(Debug, "Reallocating array of size " << m_size*sizeof(T)
                                                     << " with new size"
                                                     << new_elems*sizeof(T));

       T* new_ptr = nullptr;

       auto arrayManager = ArrayManager::getInstance();
       auto allocator = arrayManager->getAllocator(m_allocator_id);

       if (new_elems > 0) {
           new_ptr = (T*) allocator.allocate(sizeof(T) * new_elems);
           arrayManager->syncIfNeeded();
#if !defined(CHAI_THIN_GPU_ALLOCATE)
           std::memcpy(new_ptr, m_active_pointer, std::min(m_size, new_elems*sizeof(T)));
#else
           arrayManager->copy(new_ptr, m_active_pointer, std::min(m_size, new_elems*sizeof(T)));
#endif
           registerTouch(chai::GPU);
       }

       allocator.deallocate(m_active_pointer);

       m_size = new_elems*sizeof(T);
       m_active_pointer = new_ptr;
       m_active_base_pointer = m_active_pointer;

       CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
    }
  }
  else {
    CHAI_LOG(Debug, "Attempted to realloc slice!");
  }
}

template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::free(ExecutionSpace /*space*/ )
{
  if (m_allocator_id != -1) {
     if (!m_is_slice) {
       auto arrayManager = ArrayManager::getInstance();

       if (m_active_pointer) {
          auto allocator = arrayManager->getAllocator(m_allocator_id);
          allocator.deallocate((void *)m_active_pointer);
       }

       m_active_pointer = nullptr;
       m_active_base_pointer = nullptr;
       m_size = 0;
       m_allocator_id = -1;
     }
     else {
       CHAI_LOG(Debug, "tried to free slice!");
     }
  }
}


template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::reset()
{
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE typename ManagedArray<T>::T_non_const ManagedArray<
    T>::pick(size_t i) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
#if !defined(CHAI_DEVICE_COMPILE)
  ArrayManager::getInstance()->syncIfNeeded();
#endif
#elif defined(CHAI_ENABLE_UM)
  synchronize();
#endif
  return (T_non_const)m_active_pointer[i];
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T val) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
#if !defined(CHAI_DEVICE_COMPILE)
  ArrayManager::getInstance()->syncIfNeeded();
#endif
#elif defined(CHAI_ENABLE_UM)
  synchronize();
#endif
  m_active_pointer[i] = val;
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE size_t ManagedArray<T>::size() const
{
  return m_size/sizeof(T);
}

template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::registerTouch(ExecutionSpace space)
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
    chai::ArrayManager::getInstance()->setExecutionSpace(space) ;
#endif
}

template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::move(ExecutionSpace, bool) const
{
}

template <typename T>
template <typename Idx>
CHAI_INLINE CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const Idx i) const
{
  return m_active_pointer[i];
}

template <typename T>
template <typename U>
ManagedArray<T>::operator typename std::
    enable_if<!std::is_const<U>::value, ManagedArray<const U> >::type() const
{
  return ManagedArray<const T>(const_cast<const T*>(m_active_pointer),
                               m_resource_manager,
                               m_size/sizeof(T),
                               nullptr);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>&
ManagedArray<T>::operator= (ManagedArray && other) {
  if (this != &other) {
      *this = other;
      other = nullptr;
  }
  return *this;
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE ManagedArray<T>& ManagedArray<T>::operator=(std::nullptr_t from)
{
  m_active_pointer = from;
  m_active_base_pointer = from;
  m_size = 0;
  m_is_slice = false;
  m_allocator_id = -1;
  return *this;
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE bool ManagedArray<T>::operator==(
    const ManagedArray<T>& rhs) const
{
  return (m_active_pointer == rhs.m_active_pointer);
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE bool ManagedArray<T>::operator!=(
    const ManagedArray<T>& rhs) const
{
  return (m_active_pointer != rhs.m_active_pointer);
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE bool ManagedArray<T>::operator==( std::nullptr_t from) const
{
  return m_active_pointer == from;
}
template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE bool ManagedArray<T>::operator!=( std::nullptr_t from) const
{
  return m_active_pointer != from;
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE ManagedArray<T>::operator bool() const
{
  return m_active_pointer != nullptr;
}

}  // end of namespace chai

#endif  // CHAI_ManagedArray_thin_INL
