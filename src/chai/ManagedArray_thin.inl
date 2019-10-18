//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_thin_INL
#define CHAI_ManagedArray_thin_INL

#include "ManagedArray.hpp"

#if defined(CHAI_ENABLE_UM)
#include <cuda_runtime_api.h>
#endif

namespace chai {

template <typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
   std::initializer_list<chai::ExecutionSpace> spaces,
   std::initializer_list<umpire::Allocator> allocators) :
  ManagedArray()
{
  if (m_pointer_record) {
     int i = 0;

     for (auto& space : spaces) {
       m_pointer_record->m_allocators[space] = allocators.begin()[i++].getId();
     }
  }
}

template <typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems,
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators,
    ExecutionSpace space) :
  ManagedArray(spaces, allocators)
{
  m_elems = elems;
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray() = default;

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(size_t elems, ExecutionSpace space) :
  m_elems(elems)
{
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t)
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
  m_elems(elems),
  m_pointer_record(pointer_record)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST ManagedArray<T> ManagedArray<T>::slice(size_t offset, size_t elems) {
  ManagedArray<T> slice;
  if (offset + elems > size()) {
    CHAI_LOG(Debug, "Invalid slice. No active pointer or index out of bounds");
  } else {
    slice.m_active_pointer = m_active_pointer + offset;
    slice.m_elems = elems;
    slice.m_is_slice = true;
  }
  return slice;
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(size_t elems,
                                         ExecutionSpace space,
                                         UserCallback const &) {
  if (!m_is_slice) {
    (void) space; // Quiet compiler warning when CHAI_LOG does nothing
    CHAI_LOG(Debug, "Allocating array of size " << elems
                                                << " in space "
                                                << space);

    m_elems = elems;

  #if defined(CHAI_ENABLE_UM)
    cudaMallocManaged(&m_active_pointer, sizeof(T) * elems);
  #else
    m_active_pointer = static_cast<T*>(malloc(sizeof(T) * elems));
  #endif

    CHAI_LOG(Debug, "m_active_ptr allocated at address: " << m_active_pointer);
  }
  else {
    CHAI_LOG(Debug, "Attempted to allocate slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t new_elems)
{
  if (!m_is_slice) {
    CHAI_LOG(Debug, "Reallocating array of size " << m_elems
                                                  << " with new size"
                                                  << elems);

    T* new_ptr;

  #if defined(CHAI_ENABLE_UM)
    cudaMallocManaged(&new_ptr, sizeof(T) * new_elems);
    cudaMemcpy(new_ptr, m_active_pointer, sizeof(T) * m_elems, cudaMemcpyDefault);
    cudaFree(m_active_pointer);
  #else  
    new_ptr = static_cast<T*>(realloc(m_active_pointer, sizeof(T) * new_elems));
  #endif

    m_elems = new_elems;
    m_active_pointer = new_ptr;
    m_active_base_pointer = m_active_pointer;

    CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
  }
  else {
    CHAI_LOG(Debug, "Attempted to realloc slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  if (!m_is_slice) {
  #if defined(CHAI_ENABLE_UM)
    cudaFree(m_active_base_pointer);
  #else
    ::free(m_active_base_pointer);
  #endif

    m_active_base_pointer = nullptr;
    m_active_pointer = nullptr;
  }
  else {
    CHAI_LOG(Debug, "tried to free slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reset()
{
}


#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
typename ManagedArray<T>::T_non_const ManagedArray<T>::pick(size_t i) const { 
#if !defined(__CUDA_ARCH__) && defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif
  return (T_non_const) m_active_pointer[i]; 
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T& val) const { 
#if !defined(__CUDA_ARCH__) && defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif
  m_active_pointer[i] = val; 
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::incr(size_t i) const { 
#if !defined(__CUDA_ARCH__) && defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif
  ++m_active_pointer[i]; 
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::decr(size_t i) const { 
#if !defined(__CUDA_ARCH__) && defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif
  --m_active_pointer[i]; 
}
#endif

template<typename T>
CHAI_INLINE
CHAI_HOST size_t ManagedArray<T>::size() const {
  return m_elems;
}

template<typename T>
template<typename Idx>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const Idx i) const {
  return m_active_pointer[i];
}

#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::operator T*() const {
  return m_active_pointer;
}

template<typename T>
template<bool Q>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, bool test) :
  m_active_pointer(data),
  m_resource_manager(ArrayManager::getInstance()),
  m_elems(m_resource_manager->getSize(m_active_pointer)),
  m_is_slice(false)
{
}
#endif

template<typename T>
template< typename U>
ManagedArray<T>::operator 
typename std::enable_if< !std::is_const<U>::value , 
                         ManagedArray<const U> >::type () const
{
  return ManagedArray<const T>(const_cast<const T*>(m_active_pointer), m_resource_manager, m_elems, nullptr);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>&
ManagedArray<T>::operator= (std::nullptr_t from) {
  m_active_pointer = from;
  m_elems = 0;
  return *this;
}

} // end of namespace chai

#endif // CHAI_ManagedArray_thin_INL
