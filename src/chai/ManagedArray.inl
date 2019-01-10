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
  m_elems(0),
  m_offset(0),
  m_pointer_record(nullptr),
  m_is_slice(false)
{
#if !defined(__CUDA_ARCH__)
  m_resource_manager = ArrayManager::getInstance();

  m_pointer_record = new PointerRecord{};
  m_pointer_record->m_size = 0;
  m_pointer_record->m_user_callback = [](Action, ExecutionSpace, size_t) {};

  for (int space = CPU;  space < NUM_EXECUTION_SPACES; space++) {
    m_pointer_record->m_allocators[space] = 
      m_resource_manager->getAllocatorId(ExecutionSpace(space));
  }
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    std::initializer_list<chai::ExecutionSpace> spaces,
    std::initializer_list<umpire::Allocator> allocators):
  ManagedArray()
{
#if !defined(__CUDA_ARCH__)
  int i = 0;
  for (auto& space : spaces) {
    m_pointer_record->m_allocators[space] = allocators.begin()[i++].getId();
  }
#endif

}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    size_t elems, 
    ExecutionSpace space) :
  ManagedArray()
{
#if !defined(__CUDA_ARCH__)
  m_elems = elems;
  m_pointer_record->m_size = sizeof(T)*m_elems;

  this->allocate(elems, space);

#if defined(CHAI_ENABLE_UM)
  if(space == UM) {
    m_pointer_record->m_pointers[CPU] = m_active_pointer;
    m_pointer_record->m_pointers[GPU] = m_active_pointer;
  }
#endif
#endif
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
#if !defined(__CUDA_ARCH__)
  m_elems = elems;
  m_pointer_record->m_size = sizeof(T)*elems;

  this->allocate(elems, space);

  #if defined(CHAI_ENABLE_UM)
  if(space == UM) {
    m_pointer_record->m_pointers[CPU] = m_active_base_pointer;
    m_pointer_record->m_pointers[GPU] = m_active_base_pointer;
  }
  #endif
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(std::nullptr_t) :
  m_active_pointer(nullptr),
  m_active_base_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0),
  m_offset(0),
  m_pointer_record(nullptr),
  m_is_slice(false)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(PointerRecord* record, ExecutionSpace space):
  m_active_pointer(static_cast<T*>(record->m_pointers[space])),
  m_active_base_pointer(static_cast<T*>(record->m_pointers[space])),
  m_resource_manager(ArrayManager::getInstance()),
  m_elems(record->m_size/sizeof(T)),
  m_offset(0),
  m_pointer_record(record),
  m_is_slice(false)
{
}


template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_active_base_pointer(other.m_active_base_pointer),
  m_resource_manager(other.m_resource_manager),
  m_elems(other.m_elems),
  m_offset(other.m_offset),
  m_pointer_record(other.m_pointer_record),
  m_is_slice(other.m_is_slice)
{
#if !defined(__CUDA_ARCH__)
  CHAI_LOG("ManagedArray", "Moving " << m_active_pointer);
  m_active_base_pointer = static_cast<T*>(m_resource_manager->move(const_cast<T_non_const*>(m_active_base_pointer), m_pointer_record));
  m_active_pointer = m_active_base_pointer + m_offset;
  CHAI_LOG("ManagedArray", "Moved to " << m_active_pointer);

  /*
   * Register touch
   */
  if (!std::is_const<T>::value) {
    CHAI_LOG("ManagedArray", "T is non-const, registering touch of pointer" << m_active_pointer);
    m_resource_manager->registerTouch(m_pointer_record);
  }

  /// Move nested ManagedArrays
  moveInnerImpl();
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* array_manager, size_t elems, PointerRecord* pointer_record) :
  m_active_pointer(data), 
  m_active_base_pointer(data),
  m_resource_manager(array_manager),
  m_elems(elems),
  m_offset(0),
  m_pointer_record(pointer_record),
  m_is_slice(false)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST ManagedArray<T> ManagedArray<T>::slice(size_t offset, size_t elems) {
  ManagedArray<T> slice;
  slice.m_resource_manager = m_resource_manager;
  if(offset + elems > size()) {
    CHAI_LOG("ManagedArray", "Invalid slice. No active pointer or index out of bounds");
  } else {
    slice.m_pointer_record = m_pointer_record;
    slice.m_active_base_pointer = m_active_base_pointer;
    slice.m_offset = offset + m_offset;
    slice.m_active_pointer = m_active_base_pointer + slice.m_offset;
    slice.m_elems = elems;
    slice.m_is_slice = true;
  }
  return slice;
}

template<typename T>
CHAI_HOST void ManagedArray<T>::allocate(
    size_t elems,
    ExecutionSpace space, 
    const UserCallback& cback) 
{
  if(!m_is_slice) {
    CHAI_LOG("ManagedArray", "Allocating array of size " << elems << " in space " << space);

      if (space == NONE) {
        space = m_resource_manager->getDefaultAllocationSpace();
      }

  m_pointer_record->m_user_callback = cback;
  m_elems = elems;
  m_pointer_record->m_size = sizeof(T)*elems;

  m_resource_manager->allocate(m_pointer_record, space);

  m_active_base_pointer = static_cast<T*>(m_pointer_record->m_pointers[space]);
  m_active_pointer = m_active_base_pointer; // Cannot be a slice

    CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t elems)
{
  if(!m_is_slice) {
    CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << elems);

    m_elems = elems;
    m_active_base_pointer =
      static_cast<T*>(m_resource_manager->reallocate<T>(m_active_base_pointer, elems,
                                                      m_pointer_record));
    m_active_pointer = m_active_base_pointer; // Cannot be a slice

    CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << m_active_pointer);
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free(ExecutionSpace space)
{
  if(!m_is_slice) {
    m_resource_manager->free(m_pointer_record, space);
  } else {
    CHAI_LOG("ManagedArray", "Cannot free a slice!");
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
CHAI_HOST size_t ManagedArray<T>::size() const {
  return m_elems;
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::registerTouch(ExecutionSpace space) {
  m_resource_manager->registerTouch(m_pointer_record, space);
}


#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
typename ManagedArray<T>::T_non_const ManagedArray<T>::pick(size_t i) const { 
  #if !defined(__CUDA_ARCH__)
    #if defined(CHAI_ENABLE_UM)
      if(m_pointer_record->m_pointers[UM] == m_active_base_pointer) {
        cudaDeviceSynchronize();
        return (T_non_const)(m_active_pointer[i]);
      }
    #endif
    return m_resource_manager->pick(static_cast<T*>((void*)((char*)m_pointer_record->m_pointers[m_pointer_record->m_last_space]+sizeof(T)*m_offset)), i);
  #else
    return (T_non_const)(m_active_pointer[i]); 
  #endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T val) const { 
  #if !defined(__CUDA_ARCH__)
    #if defined(CHAI_ENABLE_UM)
      if(m_pointer_record->m_pointers[UM] == m_active_pointer) {
        cudaDeviceSynchronize();
        m_active_pointer[i] = val;
        return;
      }
    #endif
    m_resource_manager->set(static_cast<T*>((void*)((char*)m_pointer_record->m_pointers[m_pointer_record->m_last_space]+sizeof(T)*m_offset)), i, val);
  #else
    m_active_pointer[i] = val; 
  #endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::modify(size_t i, const T& val) const { 
  #if defined(CHAI_ENABLE_UM)
    if(m_pointer_record->m_pointers[UM] == m_active_pointer) {
      cudaDeviceSynchronize();
      m_active_pointer[i] = m_active_pointer[i] + val;
      return;
    }
  #endif
    T_non_const temp = pick(i);
    temp = temp + val;
    set(i, temp);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::incr(size_t i) const { 
  #if !defined(__CUDA_ARCH__)
    modify(i, (T)1);
  #else
     ++m_active_pointer[i]; 
  #endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::decr(size_t i) const { 
  #if !defined(__CUDA_ARCH__)
    modify(i, (T)-1);
  #else
     --m_active_pointer[i]; 
  #endif
}
#endif

template <typename T>
CHAI_INLINE
CHAI_HOST
void ManagedArray<T>::move(ExecutionSpace space)
{
  m_active_base_pointer = static_cast<T*>(m_resource_manager->move(m_active_base_pointer, m_pointer_record, space));
  m_active_pointer = m_active_base_pointer + m_offset;

  if (!std::is_const<T>::value) {
    CHAI_LOG("ManagedArray", "T is non-const, registering touch of pointer" << m_active_pointer);
    m_resource_manager->registerTouch(m_pointer_record);
  }
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
#if !defined(__CUDA_ARCH__)
  ExecutionSpace prev_space = m_resource_manager->getExecutionSpace();
  m_resource_manager->setExecutionSpace(CPU);
  auto non_const_active_base_pointer = const_cast<T_non_const*>(static_cast<T*>(m_active_base_pointer));
  m_active_base_pointer = static_cast<T_non_const*>(m_resource_manager->move(non_const_active_base_pointer, m_pointer_record));
  m_active_pointer = m_active_base_pointer;

  m_resource_manager->registerTouch(m_pointer_record);


  // Reset to whatever space we rode in on
  m_resource_manager->setExecutionSpace(prev_space);

  return m_active_pointer;
#else
  return m_active_pointer;
#endif
}


template<typename T>
template<bool Q>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, bool ) :
  m_active_pointer(data),
  m_active_base_pointer(data),
#if !defined(__CUDA_ARCH__)
  m_resource_manager(ArrayManager::getInstance()),
  m_elems(m_resource_manager->getSize(m_active_base_pointer)),
  m_pointer_record(m_resource_manager->getPointerRecord(data)),
#else
  m_resource_manager(nullptr),
  m_elems(0),
  m_pointer_record(nullptr),
#endif
  m_offset(0),
  m_is_slice(false)
{
}
#endif

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
   return m_pointer_record->m_pointers[space];
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
  return ManagedArray<const T>(const_cast<const T*>(m_active_base_pointer), 
  m_resource_manager, m_elems, m_pointer_record);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>&
ManagedArray<T>::operator= (std::nullptr_t from) {
  m_active_pointer = from;
  m_active_base_pointer = from;
  m_elems = 0;
  m_offset = 0;
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

template<typename T>
template<bool B, typename std::enable_if<B, int>::type>
CHAI_INLINE
CHAI_HOST_DEVICE
void
ManagedArray<T>::moveInnerImpl()
{
  for (int i = 0; i < size(); ++i)
  {
    auto inner = T(m_active_pointer[i]);
    m_active_pointer[i].shallowCopy(inner);
  }
}

template<typename T>
template<bool B, typename std::enable_if<!B, int>::type>
CHAI_INLINE
CHAI_HOST_DEVICE
void
ManagedArray<T>::moveInnerImpl()
{
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
