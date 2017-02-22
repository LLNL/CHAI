#ifndef CHAI_ManagedArray_CPP
#define CHAI_ManagedArray_CPP

#include "ManagedArray.hpp"

namespace chai {

template<typename T>
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_host_pointer(nullptr),
  m_device_pointer(nullptr),
  m_resource_manager(nullptr)
{
  m_resource_manager = ResourceManager::getResourceManager();
}

template<typename T>
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(size_t size):
  m_host_pointer(nullptr),
  m_device_pointer(nullptr),
  m_resource_manager(nullptr)
{
  m_resource_manager = ResourceManager::getResourceManager();
  this->allocate(size);
}

template<typename T>
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_host_pointer(other.m_host_pointer),
  m_device_pointer(other.m_device_pointer),
  m_resource_manager(other.m_resource_manager)
{
#if !defined(__CUDA_ARCH__)
  m_device_pointer = static_cast<T*>(m_resource_manager->move(other.m_host_pointer));

  /*
   * Register touch
   */

  T_non_const* non_const_pointer = static_cast<T_non_const*>(other.m_host_pointer);
  if (non_const_pointer) {
    m_resource_manager->registerTouch(non_const_pointer);
  }
#endif
}

template<typename T>
CHAI_HOST void ManagedArray<T>::allocate(size_t N) {
  m_host_pointer = static_cast<T*>(m_resource_manager->allocate<T>(N));
}

template<typename T>
CHAI_HOST size_t ManagedArray<T>::getSize() {
  return m_resource_manager->getSize(static_cast<void*>(m_host_pointer));
}

template<typename T>
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const int i) const {
#if defined(__CUDA_ARCH__)
  return m_device_pointer[i];
#else
  return m_host_pointer[i];
#endif
}

} // end namespace chai

#endif
