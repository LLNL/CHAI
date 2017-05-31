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
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_resource_manager(other.m_resource_manager),
  m_elems(other.m_elems);
{
#if !defined(__CUDA_ARCH__)
  CHAI_LOG("ManagedArray", "Moving " << m_active_pointer);

  m_active_pointer = static_cast<T*>(m_resource_manager->move(m_active_pointer));

  CHAI_LOG("ManagedArray", "Moved to " << m_active_pointer);

  /*
   * Register touch
   */
  T_non_const* non_const_pointer = static_cast<T_non_const*>(m_active_pointer);
  if (non_const_pointer) {
    m_resource_manager->registerTouch(non_const_pointer);
  }
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(uint elems, ExecutionSpace space) {
  CHAI_LOG("ManagedArray", "Allocating array of size elems in space " << space);

  m_elems = elems;
  m_active_pointer = static_cast<T*>(m_resource_manager->allocate<T>(elems, space));

  CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  m_resource_manager->free(m_active_pointer);
}


template<typename T>
CHAI_INLINE
CHAI_HOST const size_t ManagedArray<T>::size() const {
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
#if !defined(__CUDA_ARCH__)
  m_resource_manager->setExecutionSpace(CPU);

  m_active_pointer = static_cast<T*>(m_resource_manager->move(m_active_pointer));

  m_resource_manager->registerTouch(m_active_pointer);

  return m_active_pointer;
#else
  return m_active_pointer;
#endif
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
