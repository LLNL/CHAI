// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
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
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0),
  m_user_callback([](Action, ExecutionSpace, size_t){})
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems, ExecutionSpace space):
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(elems),
  m_user_callback([](Action, ExecutionSpace, size_t){})
{
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t) :
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0),
  m_user_callback([](Action, ExecutionSpace, size_t){})
{
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_resource_manager(other.m_resource_manager),
  m_elems(other.m_elems),
  m_user_callback([](Action, ExecutionSpace, size_t){})
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* array_manager, size_t elems) :
  m_active_pointer(data), 
  m_resource_manager(array_manager),
  m_elems(elems),
  m_user_callback([](Action, ExecutionSpace, size_t){})
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(size_t elems, ExecutionSpace space, 
    UserCallback const &cback) {
  CHAI_LOG("ManagedArray", "Allocating array of size " << elems << " in space " << space);

  m_elems = elems;
  m_user_callback = cback;

#if defined(CHAI_ENABLE_UM)
  m_user_callback(ACTION_ALLOC, UM, sizeof(T)*elems); 
  cudaMallocManaged(&m_active_pointer, sizeof(T)*elems);
#else
  m_user_callback(ACTION_ALLOC, CPU, sizeof(T)*elems); 
  m_active_pointer = static_cast<T*>(malloc(sizeof(T)*elems));
#endif

  CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t new_elems)
{
  CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << elems);

  T* new_ptr;
#if defined(CHAI_ENABLE_UM)
  m_user_callback(ACTION_FREE, UM, sizeof(T)*m_elems); 
  m_user_callback(ACTION_ALLOC, UM, sizeof(T)*new_elems); 
  
  cudaMallocManaged(&new_ptr, sizeof(T)*new_elems);

  cudaMemcpy(new_ptr, m_active_pointer, sizeof(T)*m_elems, cudaMemcpyDefault);

  cudaFree(m_active_pointer);
#else
  m_user_callback(ACTION_FREE, CPU, sizeof(T)*m_elems); 
  m_user_callback(ACTION_ALLOC, CPU, sizeof(T)*new_elems); 
  
  new_ptr = static_cast<T*>(realloc(m_active_pointer, sizeof(T)*new_elems));
#endif

  m_elems = new_elems;
  m_active_pointer = new_ptr;

  CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  
#if defined(CHAI_ENABLE_UM)
  m_user_callback(ACTION_FREE, UM, sizeof(T)*m_elems); 
  cudaFree(m_active_pointer);
#else
  m_user_callback(ACTION_FREE, CPU, sizeof(T)*m_elems);
  ::free(m_active_pointer);
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reset()
{
}


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
  m_elems(m_resource_manager->getSize(m_active_pointer))
{
}
#endif

template<typename T>
template<bool B,typename std::enable_if<!B, int>::type>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>::operator ManagedArray<const T> () const
{
  return ManagedArray<const T>(const_cast<const T*>(m_active_pointer), m_resource_manager, m_elems);
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


template<typename T>
CHAI_HOST void ManagedArray<T>::setUserCallback(UserCallback const &cback)
{
  m_user_callback = cback;  
}



} // end of namespace chai

#endif // CHAI_ManagedArray_thin_INL
