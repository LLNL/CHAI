// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
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
#ifndef CHAI_ManagedArray_thin_INL
#define CHAI_ManagedArray_thin_INL

#include "ManagedArray.hpp"

#if defined(CHAI_ENABLE_UM)
#include <cuda_runtime_api.h>
#endif

namespace chai {


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators):
  ManagedArray()
{
  int i = 0;
  for (auto& space : spaces) {
    m_pointer_record->m_allocators[space] = allocators.begin()[i++].getId();
  }
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems, 
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators,
    ExecutionSpace space):
  ManagedArray(spaces, allocators)
{
  m_elems = elems;
  this->allocate(elems, space);
}



template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0),
  m_is_slice(false)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems, ExecutionSpace space):
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_pointer_record(nullptr),
  m_elems(elems),
  m_is_slice(false)
{
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t) :
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0),
  m_is_slice(false)
{
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_active_base_pointer(other.m_active_base_pointer),
  m_resource_manager(other.m_resource_manager),
  m_pointer_record(other.m_pointer_record),
  m_elems(other.m_elems),
  m_is_slice(other.m_is_slice)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* , size_t elems, PointerRecord* ) :
  m_active_pointer(data),
  m_active_base_pointer(data),
  m_resource_manager(nullptr),
  m_pointer_record(nullptr),
  m_elems(elems),
  m_is_slice(false)
{
}

template<typename T>
T*
ManagedArray<T>::getActiveBasePointer() const
{
  return m_active_base_pointer;
}

template<typename T>
T*
ManagedArray<T>::getActivePointer() const
{
  return m_active_pointer;
}

template<typename T> 
T*
ManagedArray<T>::getPointer(ExecutionSpace space) const { 
   return m_active_pointer;
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(size_t elems, ExecutionSpace space, UserCallback const &cback) {
  if(!m_is_slice) {
    //CHAI_LOG("ManagedArray", "Allocating array of size " << elems << " in space " << space);

    m_elems = elems;
    if (m_elems != 0 ) {

     #if defined(CHAI_ENABLE_UM)
       cudaMallocManaged(&m_active_pointer, sizeof(T)*elems);
     #else
       m_active_pointer = static_cast<T*>(malloc(sizeof(T)*elems));
     #endif
    }
    else {
       m_active_pointer = nullptr;
    }
    m_active_base_pointer = m_active_pointer;

    //CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << (void *)m_active_pointer);
  }
  else {
     CHAI_LOG("ManagedArray", "Attempted to allocate slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t new_elems)
{
  if(!m_is_slice) {
    CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << new_elems);

    T* new_ptr;
  #if defined(CHAI_ENABLE_UM)
    if (new_elems != 0) { 
       cudaMallocManaged(&new_ptr, sizeof(T)*new_elems);

       cudaMemcpy(new_ptr, m_active_pointer, sizeof(T)*m_elems, cudaMemcpyDefault);
       
    } else {
       new_ptr = nullptr;
    }

    cudaFree(m_active_pointer);
  #else  
    if (new_elems == 0) { 
       ::free(m_active_pointer);
       new_ptr = nullptr;
    } else {
       new_ptr = static_cast<T*>(realloc(m_active_pointer, sizeof(T)*new_elems));
    }
  #endif

    m_elems = new_elems;
    m_active_pointer = new_ptr;
    m_active_base_pointer = m_active_pointer;

    CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << (void *) m_active_pointer);
  }
  else {
    CHAI_LOG("ManagedArray", "Attempted to realloc slice!");
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free(ExecutionSpace space)
{
   if (space == CPU || space == NONE) {
      if(!m_is_slice) {
      #if defined(CHAI_ENABLE_UM)
        cudaFree(m_active_pointer);
      #else
        ::free(m_active_pointer);
      #endif
      m_active_pointer = nullptr;
      m_active_base_pointer = nullptr;
      }
      else {
         CHAI_LOG("ManagedArray", "tried to free slice!");
      }
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
CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T val) const { 
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
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::registerTouch(ExecutionSpace ) {
}

template <typename T>
CHAI_INLINE
CHAI_HOST
void ManagedArray<T>::move(ExecutionSpace ) {
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
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, CHAIDISAMBIGUATE name, bool test) :
  m_active_pointer(data), m_active_base_pointer(data), m_resource_manager(nullptr), m_pointer_record(nullptr), m_elems(-1), m_is_slice(false)
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
  m_active_base_pointer = from;
  m_elems = 0;
  return *this;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator== (ManagedArray<T>& rhs)
{
  return (m_active_pointer ==  rhs.m_active_pointer);
}

} // end of namespace chai

#endif // CHAI_ManagedArray_thin_INL
