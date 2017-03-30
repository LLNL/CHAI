#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/ChaiMacros.hpp"
#include "chai/ArrayManager.hpp"

#include "chai/Types.hpp"

namespace chai {

template <typename T>
class ManagedArray {
  public:

  using T_non_const = typename std::remove_const<T>::type;

  CHAI_HOST_DEVICE ManagedArray();

  CHAI_HOST_DEVICE ManagedArray(uint elems, ExecutionSpace space=CPU);

  CHAI_HOST_DEVICE ManagedArray(ManagedArray const& other);

  CHAI_HOST void allocate(uint elems, ExecutionSpace space=CPU);
  CHAI_HOST size_t getSize();

  CHAI_HOST_DEVICE T& operator[](const int i) const;

  CHAI_HOST_DEVICE operator T*() const;

  private:

  mutable T* m_host_pointer;
  mutable T* m_device_pointer;

  ArrayManager* m_resource_manager;
};

}

#include "chai/ManagedArray.inl"

#endif
