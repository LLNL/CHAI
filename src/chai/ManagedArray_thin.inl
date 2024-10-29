//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
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
  if (m_pointer_record) {
     int i = 0;

     for (const auto& space : spaces) {
       m_pointer_record->m_allocators[space] = allocators.begin()[i++].getId();
     }
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

template<typename T>
CHAI_INLINE
CHAI_HOST ManagedArray<T>::ManagedArray(PointerRecord* record, ExecutionSpace space):
  m_active_pointer(static_cast<T*>(record->m_pointers[space])),
  m_active_base_pointer(static_cast<T*>(record->m_pointers[space])),
  m_resource_manager(nullptr),
  m_size(record->m_size),
  m_offset(0),
  m_pointer_record(nullptr),
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
  m_pointer_record(pointer_record)
{
}

template <typename T>
CHAI_HOST_DEVICE T* ManagedArray<T>::getActiveBasePointer() const
{
  return m_active_base_pointer;
}

template <typename T>
CHAI_HOST_DEVICE T* ManagedArray<T>::getActivePointer() const
{
  return m_active_pointer;
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
     if (elems > 0) {
       (void) space; // Quiet compiler warning when CHAI_LOG does nothing
       CHAI_LOG(Debug, "Allocating array of size " << elems
                                                   << " in space "
                                                   << space);

       m_size = elems*sizeof(T);

     #if defined(CHAI_THIN_GPU_ALLOCATE)
       m_active_pointer = (T*) chai::ArrayManager::getInstance()->getAllocator(chai::GPU).allocate(m_size);
     #elif defined(CHAI_ENABLE_UM)
       gpuMallocManaged(&m_active_pointer, m_size);
     #else // not CHAI_ENABLE_UM
       m_active_pointer = static_cast<T*>(malloc(sizeof(T) * elems));
     #endif

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
    CHAI_LOG(Debug, "Reallocating array of size " << m_size*sizeof(T)
                                                  << " with new size"
                                                  << new_elems*sizeof(T));

    T* new_ptr = nullptr;

  #if defined(CHAI_THIN_GPU_ALLOCATE)
    auto allocator = chai::ArrayManager::getInstance()->getAllocator(chai::GPU);
    if (new_elems > 0) {
        new_ptr = (T*) allocator.allocate(sizeof(T) * new_elems);
        ArrayManager::getInstance()->syncIfNeeded();
        chai::gpuMemcpy(new_ptr, m_active_pointer, std::min(m_size, new_elems*sizeof(T)), gpuMemcpyDefault);
        registerTouch(chai::GPU);
    }
    allocator.deallocate(m_active_pointer);
  #elif defined(CHAI_ENABLE_UM)
    if (new_elems > 0) {
       gpuMallocManaged(&new_ptr, sizeof(T) * new_elems);
       gpuMemcpy(new_ptr, m_active_pointer, std::min(new_elems*sizeof(T), m_size), gpuMemcpyDefault);
    }
    gpuFree(m_active_pointer);
  #else  // not CHAI_ENABLE_UM
    if (new_elems > 0) {
       new_ptr = static_cast<T*>(realloc(m_active_pointer, sizeof(T) * new_elems));
    }
    else {
       ::free((void *)m_active_pointer);
    }
  #endif

    m_size= new_elems*sizeof(T);
    m_active_pointer = new_ptr;
    m_active_base_pointer = m_active_pointer;

    CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
  }
  else {
    CHAI_LOG(Debug, "Attempted to realloc slice!");
  }
}

template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::free(ExecutionSpace space)
{
  if (!m_is_slice) {
    if (space == CPU || space == NONE) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
      if (m_active_pointer) {
         auto allocator = chai::ArrayManager::getInstance()->getAllocator(chai::GPU);
         allocator.deallocate((void *)m_active_pointer);
      }
#elif defined(CHAI_ENABLE_UM)
      chai::gpuFree(m_active_pointer);
#else
      ::free((void *)m_active_pointer);
#endif
      m_active_pointer = nullptr;
      m_active_base_pointer = nullptr;
      m_size = 0;
    }
  }
  else {
    CHAI_LOG(Debug, "tried to free slice!");
  }
}


template <typename T>
CHAI_INLINE CHAI_HOST void ManagedArray<T>::reset()
{
}


#if defined(CHAI_ENABLE_PICK)
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
CHAI_INLINE CHAI_HOST_DEVICE void ManagedArray<T>::incr(size_t i) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
#if !defined(CHAI_DEVICE_COMPILE)
  ArrayManager::getInstance()->syncIfNeeded();
#endif
#elif defined(CHAI_ENABLE_UM)
   synchronize();
#endif
  ++m_active_pointer[i];
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE void ManagedArray<T>::decr(size_t i) const
{
#if defined(CHAI_THIN_GPU_ALLOCATE)
#if !defined(CHAI_DEVICE_COMPILE)
  ArrayManager::getInstance()->syncIfNeeded();
#endif
#elif defined(CHAI_ENABLE_UM)
   synchronize();
#endif
  --m_active_pointer[i];
}
#endif // CHAI_ENABLE_PICK

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
CHAI_INLINE CHAI_HOST_DEVICE ManagedArray<T>::operator bool() const
{
  return m_active_pointer != nullptr;
}

}  // end of namespace chai

#endif  // CHAI_ManagedArray_thin_INL
