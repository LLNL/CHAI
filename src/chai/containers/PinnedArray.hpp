//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_PINNED_ARRAY_HPP
#define CHAI_PINNED_ARRAY_HPP

#include "chai/config.hpp"
#include "camp/Resource.hpp"
#include "hip/hip_runtime_api.h"
#include <cstddef>

namespace chai {
namespace experimental {

template <class T>
class PinnedAllocator
{
  T* allocate(std::size_t count, camp::resources::Hip resource)
  {
    T* pointer;
    hipMallocAsync(&pointer, count*sizeof(T), resource.get_stream());
    return pointer;
  }

  void deallocate(T* pointer, camp::resources::Hip resource)
  {
    hipFreeAsync(pointer, resource.get_stream());
  }
};

template <class T, class Allocator = PinnedAllocator>
class PinnedArray
{
public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  using resource_type = camp::resources::Hip;
  using T_non_const = typename std::remove_const<T>::type;

  PinnedArray() = default;

  explicit PinnedArray(size_type count,
                       resource_type resource = resource_type::get_default(),
                       const allocator_type& allocator = allocator_type()) :
    m_control{new ControlBlock()}
    m_size{count},
    m_data{allocator.allocate(count, resource)}
  {
    m_control->allocator = allocator;
    m_control->resource = resource;
    m_control->event = resource.get_event();
  }

  PinnedArray(const PinnedArray& other) = default;

  PinnedArray& operator=(PinnedArray&& other) = default;

  void resize(size_type n) {
    if (m_size != n) {
      if (n == 0) {
         free();
      }
      else if (m_size == 0) {
         // Allocate in the default space?
         m_data = static_cast<T*>(m_allocator.allocate(n));
         m_size = n;
      }
      else {
        size_type copyCount = std::min(m_size, n);

        T* new_data = static_cast<T*>(m_allocator.allocate(n));
        // copy to new data
        m_allocator.deallocate(m_data);
        m_data = new_data;
        m_size = n;
      }
    }
  }

  void free() {
    m_allocator.deallocate(m_data);
    m_data = nullptr;
    m_size = 0;
  }

  CHAI_HOST_DEVICE size_type size() const {
    return m_size;
  }

  CHAI_HOST_DEVICE reference operator[](size_type i) const {
    return m_storage[i];
  }

private:
  struct ControlBlock {
     Allocator m_allocator;
     camp::resources::Resource m_last_resource;
     camp::resources::Event m_last_event;
  };

  size_type m_size = 0;  //!< Number of elements
  pointer m_data = nullptr;  //!< Pointer to data
  ControlBlock* m_control = nullptr;  //!< Pointer to control block

};  // class PinnedArray

}  // namespace experimental
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_HPP
