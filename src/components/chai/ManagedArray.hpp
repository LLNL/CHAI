#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/ChaiMacros.hpp"
#include "chai/ResourceManager.hpp"

namespace chai {

template <typename T>
class ManagedArray {
  public:

  using T_non_const = typename std::remove_const<T>::type;

  CHAI_HOST_DEVICE  ManagedArray();

  CHAI_HOST_DEVICE ManagedArray(size_t size);

  CHAI_HOST_DEVICE ManagedArray(size_t size, std::string location="default");

  CHAI_HOST_DEVICE ManagedArray(ManagedArray const& other);

  CHAI_HOST void allocate(size_t N);
  CHAI_HOST size_t getSize();

  CHAI_HOST_DEVICE T& operator[](const int i) const;

  private:

  mutable T* m_host_pointer;
  mutable T* m_device_pointer;

  ResourceManager* m_resource_manager;
};

}

#include "chai/ManagedArray.inl"

#endif
