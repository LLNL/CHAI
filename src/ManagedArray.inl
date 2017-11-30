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
#ifndef CHAI_ManagedArray_INL
#define CHAI_ManagedArray_INL

#include "ManagedArray.hpp"
#include "ArrayManager.hpp"

namespace chai {

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0)
{
  m_resource_manager = ArrayManager::getInstance();
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    uint elems, ExecutionSpace space):
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(elems)
{
  m_resource_manager = ArrayManager::getInstance();
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t) :
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0)
{
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_resource_manager(other.m_resource_manager),
  m_elems(other.m_elems)
{
#if !defined(__CUDA_ARCH__)
  CHAI_LOG("ManagedArray", "Moving " << m_active_pointer);

  m_active_pointer = static_cast<T*>(m_resource_manager->move(const_cast<T_non_const*>(m_active_pointer)));

  CHAI_LOG("ManagedArray", "Moved to " << m_active_pointer);

  /*
   * Register touch
   */
  if (!std::is_const<T>::value) {
    CHAI_LOG("ManagedArray", "T is non-const, registering touch of pointer" << m_active_pointer);
    T_non_const* non_const_pointer = const_cast<T_non_const*>(m_active_pointer);
    m_resource_manager->registerTouch(non_const_pointer);
  }
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* array_manager, uint elems) :
  m_active_pointer(data), 
  m_resource_manager(array_manager),
  m_elems(elems)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(uint elems, ExecutionSpace space) {
  CHAI_LOG("ManagedArray", "Allocating array of size " << elems << " in space " << space);

  if (space == NONE) {
    space = m_resource_manager->getDefaultAllocationSpace();
  }

  m_elems = elems;
  m_active_pointer = static_cast<T*>(m_resource_manager->allocate<T>(elems, space));

  CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(uint elems)
{
  CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << elems);

  m_elems = elems;
  m_active_pointer = static_cast<T*>(m_resource_manager->reallocate<T>(m_active_pointer, elems));

  CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  m_resource_manager->free(m_active_pointer);
}


template<typename T>
CHAI_INLINE
CHAI_HOST uint ManagedArray<T>::size() const {
  return m_elems;
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::registerTouch(ExecutionSpace space) {
  m_resource_manager->registerTouch(m_active_pointer, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const int i) const {
  return m_active_pointer[i];
}

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

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
