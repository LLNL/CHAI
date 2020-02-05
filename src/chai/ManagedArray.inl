//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
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
  m_elems(0),
  m_offset(0),
  m_pointer_record(nullptr),
  m_is_slice(false)
{
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  move(m_resource_manager->getExecutionSpace(), m_resource_manager->getResource());
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
  ManagedArray<T> slice(nullptr);
  slice.m_resource_manager = m_resource_manager;
  if(offset + elems > size()) {
    CHAI_LOG(Debug, "Invalid slice. No active pointer or index out of bounds");
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
    CHAI_LOG(Debug, "Allocating array of size " << elems << " in space " << space);

    if (space == NONE) {
      space = m_resource_manager->getDefaultAllocationSpace();
    }

    setUserCallback(cback);
    m_elems = elems;
    m_pointer_record->m_size = sizeof(T)*elems;

    m_resource_manager->allocate(m_pointer_record, space);

    m_active_base_pointer = static_cast<T*>(m_pointer_record->m_pointers[space]);
    m_active_pointer = m_active_base_pointer; // Cannot be a slice

    CHAI_LOG(Debug, "m_active_ptr allocated at address: " << m_active_pointer);
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(size_t elems)
{
  if(!m_is_slice) {
    CHAI_LOG(Debug, "Reallocating array of size " << m_elems << " with new size" << elems);

    m_elems = elems;
    m_active_base_pointer =
      static_cast<T*>(m_resource_manager->reallocate<T>(m_active_base_pointer, elems,
                                                      m_pointer_record));
    m_active_pointer = m_active_base_pointer; // Cannot be a slice

    CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
  }
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  if(!m_is_slice) {
    m_resource_manager->free(m_pointer_record);
    m_pointer_record = nullptr;
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
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
CHAI_HOST_DEVICE void ManagedArray<T>::set(size_t i, T& val) const { 
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    modify(i, (T)1);
  #else
     ++m_active_pointer[i]; 
  #endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE void ManagedArray<T>::decr(size_t i) const { 
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
  ExecutionSpace prev_space = m_pointer_record->m_last_space;

  /* When moving from CPU to GPU we need to move the inner arrays before the outer array. */
  if (prev_space == CPU) {
    moveInnerImpl(space);
  }

  m_active_base_pointer = static_cast<T*>(m_resource_manager->move(const_cast<T_non_const*>(m_active_base_pointer), m_pointer_record, space));
  m_active_pointer = m_active_base_pointer + m_offset;

  if (!std::is_const<T>::value) {
    CHAI_LOG(Debug, "T is non-const, registering touch of pointer" << m_active_pointer);
    m_resource_manager->registerTouch(m_pointer_record, space);
  }

  if (space != NONE) m_pointer_record->m_last_space = space;

  /* When moving from GPU to CPU we need to move the inner arrays after the outer array. */
#if defined(CHAI_ENABLE_CUDA)
  if (prev_space == GPU) {
    moveInnerImpl(space);
  }
#endif
}
template <typename T>
CHAI_INLINE
CHAI_HOST
void ManagedArray<T>::move(ExecutionSpace space, camp::resources::Resource* resource)
{
  ExecutionSpace prev_space = m_pointer_record->m_last_space;

  /* When moving from CPU to GPU we need to move the inner arrays before the outer array. */
  if (prev_space == CPU) {
    moveInnerImpl(space);
  }

  m_active_base_pointer = static_cast<T*>(m_resource_manager->move(const_cast<T_non_const*>(m_active_base_pointer), m_pointer_record, resource, space));
  m_active_pointer = m_active_base_pointer + m_offset;

  if (!std::is_const<T>::value) {
    CHAI_LOG(Debug, "T is non-const, registering touch of pointer" << m_active_pointer);
    m_resource_manager->registerTouch(m_pointer_record, space);
  }

  if (space != NONE) m_pointer_record->m_last_space = space;
  if (space != NONE) m_pointer_record->m_last_resource = resource;

  /* When moving from GPU to CPU we need to move the inner arrays after the outer array. */
#if defined(CHAI_ENABLE_CUDA)
  if (prev_space == GPU) {
    moveInnerImpl(space);
  }
#endif
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
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
  m_elems = 0;
  m_offset = 0;
  m_pointer_record = nullptr;
  m_is_slice = false;
  return *this;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE
bool
ManagedArray<T>::operator== (ManagedArray<T>& rhs) {
  return (m_active_pointer ==  rhs.m_active_pointer);
}

template<typename T>
template<bool B, typename std::enable_if<B, int>::type>
CHAI_INLINE
CHAI_HOST
void
ManagedArray<T>::moveInnerImpl(ExecutionSpace space) {
  if (space == NONE) {
    return;
  }

  ExecutionSpace const prev_space = m_resource_manager->getExecutionSpace();
  m_resource_manager->setExecutionSpace(space);

  T_non_const* non_const_active_base_pointer = const_cast<T_non_const*>(m_active_base_pointer);
  for (int i = 0; i < size(); ++i) {
    non_const_active_base_pointer[i] = T(m_active_base_pointer[i]);
  }

  m_resource_manager->setExecutionSpace(prev_space);
}

template<typename T>
template<bool B, typename std::enable_if<!B, int>::type>
CHAI_INLINE
CHAI_HOST
void
ManagedArray<T>::moveInnerImpl(ExecutionSpace CHAI_UNUSED_ARG(space))
{
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
