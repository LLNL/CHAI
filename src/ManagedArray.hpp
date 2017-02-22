#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

namespace chai {

class ManagedArray {

  CHAI_HOST_DEVICE  ManagedArray():
    m_pointer(nullptr)
  {
  }

  CHAI_HOST_DEVICE ManagedArray(size_t size) {
  }

  CHAI_HOST size_t getSize() {
    return m_resource_manager->getSize(static_cast<void*>(m_host_ptr));
  }

  CHAI_HOST_DEVICE operator T*() const {
  }

  CHAI_HOST_DEVICE T& operator[](const int i) {
#ifdef __CUDA_ARCH__
    return m_active_ptr[i];
#else
  }


  private:

  mutable T* m_pointer;
  ExecutionSpace current_space;

  ResourceManager* m_resource_manager;
};

}

#endif
