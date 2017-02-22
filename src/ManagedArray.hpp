#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "ChaiMacros.hpp"

#include "ResourceManager.hpp"

namespace chai {

template <typename T>
class ManagedArray {
  public:

  CHAI_HOST_DEVICE  ManagedArray():
    m_host_pointer(nullptr),
    m_device_pointer(nullptr),
    m_resource_manager(nullptr)
  {
  }

  CHAI_HOST_DEVICE ManagedArray(size_t size):
    m_host_pointer(nullptr),
    m_device_pointer(nullptr),
    m_resource_manager(nullptr)
  {
    m_resource_manager = ResourceManager::getResourceManager();
    this->allocate(size);
  }

  CHAI_HOST_DEVICE ManagedArray(ManagedArray const& other):
    m_host_pointer(other.m_host_pointer),
    m_device_pointer(other.m_device_pointer),
    m_resource_manager(other.m_resource_manager)
  {
#if !defined(__CUDA_ARCH__)
    m_device_pointer = static_cast<T*>(m_resource_manager->move(other.m_host_pointer));

    /*
     * Register touch
     */

    // un_cost* register_touch = static_cast<un_const*>(other.m_host_pointer);
    // if (register_touch) {
      m_resource_manager->registerTouch(other.m_host_pointer);
    //}
#endif
  }

  CHAI_HOST void allocate(size_t N) {
    m_host_pointer = static_cast<T*>(m_resource_manager->allocate<T>(N));
  }

  CHAI_HOST size_t getSize() {
    return m_resource_manager->getSize(static_cast<void*>(m_host_pointer));
  }

  CHAI_HOST_DEVICE T& operator[](const int i) const {
#if defined(__CUDA_ARCH__)
    return m_device_pointer[i];
#else
    return m_host_pointer[i];
#endif
  }

  private:

  mutable T* m_host_pointer;
  mutable T* m_device_pointer;

  ResourceManager* m_resource_manager;
};

}

#include "ManagedArray.cpp"

#endif
