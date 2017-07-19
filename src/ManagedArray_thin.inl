#ifndef CHAI_ManagedArray_thin_INL
#define CHAI_ManagedArray_thin_INL

#include "ManagedArray.hpp"

#include <cuda_runtime_api.h>

namespace chai {

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    uint elems, ExecutionSpace space):
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(elems)
{
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

  m_elems = elems;
  cudaMallocManaged(&m_active_pointer, sizeof(T)*elems);

  CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(uint new_elems)
{
  CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << elems);

  T* new_ptr;
  cudaMallocManaged(&new_ptr, sizeof(T)*new_elems);

  cudaMemcpy(new_ptr, m_active_pointer, sizeof(T)*m_elems, cudaMemcpyDefault);

  cudaFree(m_active_pointer);

  m_elems = new_elems;
  m_active_pointer = new_ptr;

  CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  cudaFree(m_active_pointer);
}


template<typename T>
CHAI_INLINE
CHAI_HOST uint ManagedArray<T>::size() const {
  return m_elems;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const int i) const {
  return m_active_pointer[i];
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::operator T*() const {
  return m_active_pointer;
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

#endif // CHAI_ManagedArray_thin_INL
