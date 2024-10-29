//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_HOST_ARRAY_HPP
#define CHAI_HOST_ARRAY_HPP

namespace chai {
namespace experimental {

template <class T, class Allocator>
class HostArray
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

  using T_non_const = typename std::remove_const<T>::type;

  HostArray() = default;

  explicit HostArray(const Allocator& alloc);

  explicit HostArray(size_type count, const Allocator& alloc = Allocator())
    : m_count(count*sizeof(T))
  {
    this->allocate(elems, space);
  }

  HostArray(const HostArray& other) = default;

  T* data(ExecutionSpace space, bool do_move) const
  {
    return m_active_pointer;
  }

  T* getPointer(ExecutionSpace space, bool do_move) const
  {
    return data(space, do_move);
  }

  T* begin() const {
    return data();
  }

  T* end() const {
    return data() + size();
  }

  void allocate(size_t elems, ExecutionSpace space,
                const UserCallback& cback = [] (const PointerRecord*, Action, ExecutionSpace) {}) {
    if (!m_is_slice) {
      if (elems > 0) {
        (void) space; // Quiet compiler warning when CHAI_LOG does nothing
        CHAI_LOG(Debug, "Allocating array of size " << elems
                                                    << " in space "
                                                    << space);

        m_size = elems*sizeof(T);
        m_active_pointer = static_cast<T*>(::malloc(sizeof(T) * elems));

        CHAI_LOG(Debug, "m_active_ptr allocated at address: " << m_active_pointer);
      }
      else {
        m_size = 0;
        m_active_pointer = nullptr;
      }
    }
    else {
      CHAI_LOG(Debug, "Attempted to allocate slice!");
    }

    m_active_base_pointer = m_active_pointer;
  }

  void reallocate(size_t new_elems) {
    if (!m_is_slice) {
      CHAI_LOG(Debug, "Reallocating array of size " << m_size*sizeof(T)
                                                    << " with new size"
                                                    << new_elems*sizeof(T));

      T* new_ptr = nullptr;

      if (new_elems > 0) {
        new_ptr = static_cast<T*>(::realloc(m_active_pointer, sizeof(T) * new_elems));
      }
      else {
        ::free((void *)m_active_pointer);
      }

      m_size = new_elems*sizeof(T);
      m_active_pointer = new_ptr;
      m_active_base_pointer = m_active_pointer;

      CHAI_LOG(Debug, "m_active_ptr reallocated at address: " << m_active_pointer);
    }
    else {
      CHAI_LOG(Debug, "Attempted to realloc slice!");
    }
  }

  void free(ExecutionSpace space = NONE) {
    if (!m_is_slice) {
      if (space == CPU || space == NONE) {
        ::free((void *)m_active_pointer);
        m_size = 0;
        m_active_pointer = nullptr;
        m_active_base_pointer = nullptr;
      }
    }
    else {
      CHAI_LOG(Debug, "Attempted to free slice!");
    }
  }

  void reset() {}

#if defined(CHAI_ENABLE_PICK)
  typename T_non_const pick(size_t i) const {
    return m_active_pointer[i];
  }

  void set(size_t i, T_non_const val) const {
    m_active_pointer[i] = val;
  }

  void incr(size_t i) const {
    ++m_active_pointer[i];
  }
  
  void decr(size_t i) const {
    --m_active_pointer[i];
  }
#endif // CHAI_ENABLE_PICK

  size_t size() const {
    return m_size / sizeof(T);
  }

  void registerTouch(ExecutionSpace space) {}

  void move(ExecutionSpace, bool) const {}

  reference operator[](size_type i) const {
    return m_storage[i];
  }

  pointer data() const {
    return m_storage;
  }

  const_pointer cdata() const {
    return m_storage;
  }





  template <typename U>
  operator typename std::enable_if<!std::is_const<U>::value, HostArray<const U> >::type() const
  {
    return HostArray<const T>(const_cast<const T*>(m_active_pointer),
                              m_resource_manager,
                              m_size/sizeof(T),
                              nullptr);
  }

  HostArray& operator=(HostArray&& other) {
    if (this != &other) {
      *this = other;
      other = nullptr;
    }

    return *this;
  }

  operator bool() const {
    return m_bytes > 0;
  }

private:
  size_type m_bytes = 0;  //!< Number of bytes
  pointer m_storage = nullptr;  //!< Pointer to data
};  // class HostArray

}  // namespace experimental
}  // namespace chai

#endif  // CHAI_HOST_ARRAY_HPP
