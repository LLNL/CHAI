
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPointerCounter_HPP
#define CHAI_SharedPointerCounter_HPP

#include <initializer_list>
#include <type_traits>
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/SharedPtrManager.hpp"

namespace chai
{
namespace expt
{

/*
 * The base type for shared ptr counter types.
 *
 * msp_couted_base is responsible for managing the live count of the owned
 * object. It is also responsible for defining the behavior when a release of
 * the owned object should be triggered.
 */
class msp_counted_base {
public:
  msp_counted_base() noexcept : m_use_count(1) {}

  virtual ~msp_counted_base() noexcept {}

  virtual void m_dispose() noexcept = 0;
  virtual void m_destroy() noexcept { delete this; }

  virtual void moveInnerImpl() const = 0;

  void m_add_ref_copy() noexcept { ++m_use_count; }

  void m_release() noexcept {
    if(--m_use_count == 0) {
      m_dispose();
      m_destroy();
    }
  }

  long m_get_use_count() const noexcept { return m_use_count; }

  virtual msp_pointer_record* m_get_record() noexcept = 0;

private:
  msp_counted_base(msp_counted_base const&) = delete;
  msp_counted_base& operator=(msp_counted_base const&) = delete;

  long m_use_count = 0;
};

/*
 * The default counter type for a shared ptr.
 *
 * msp_counted_ptr maintains the current pointer record and implements the
 * underlying deallocation behavior for all execution spaces upon release of the
 * owned object.
 */
template<typename Ptr>
class msp_counted_ptr final : public msp_counted_base {
public:
  msp_counted_ptr(Ptr h_p, Ptr d_p) noexcept 
    : m_record(SharedPtrManager::getInstance()->makeSharedPtrRecord(h_p, d_p, sizeof(std::remove_pointer<Ptr>), true)) 
  {}

  virtual void m_dispose() noexcept { 

    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      Ptr ptr = (Ptr)m_record->m_pointers[space];
      if (ptr) {
        SharedPtrManager::getInstance()->free(m_record, ExecutionSpace(space));
      }
    }
  }
  virtual void m_destroy() noexcept { delete this; }

  virtual void moveInnerImpl() const {
    using T = std::remove_pointer_t<Ptr>;
    Ptr host_ptr = (Ptr) m_record->m_pointers[CPU]; 
    // trigger the copy constructor
    T inner = T(*host_ptr);
    // ensure the inner type gets the state of the result of the copy
    host_ptr->operator=(inner);
  }

  msp_counted_ptr(msp_counted_ptr const&) = delete;
  msp_counted_ptr& operator=(msp_counted_ptr const&) = delete;
  msp_pointer_record* m_get_record() noexcept { return m_record; }
private:
  msp_pointer_record* m_record = nullptr;
};

#include <typeinfo>

#if defined(CHAI_GPUCC)
namespace impl {

template <typename T,
          typename Deleter>
__global__ void msp_dispose_on_device(T* gpuPointer, Deleter d)
{
   d(gpuPointer);
}

} // namespace impl
#endif

/*
 * The counter type when users have defined explicit deleters for an owned type.
 *
 * msp_counted_deleter maintains the current pointer record and calls the
 * user defined deleter for all execution spaces upon release of the
 * owned object.
 */
template<typename Ptr, typename Deleter>
class msp_counted_deleter final : public msp_counted_base {

  class impl {
  public:
    template<typename Ptrs, typename Spaces>
    impl(Ptrs&& ptrs, Spaces&& spaces, Deleter d) 
      : m_record(SharedPtrManager::getInstance()->
          makeSharedPtrRecord(std::forward<Ptrs>(ptrs), std::forward<Spaces>(spaces), 
                              sizeof(std::remove_pointer_t<Ptr>), true))
      , m_deleter(std::move(d)) 
    {}
    ~impl() { if (m_record) delete m_record; }

    Deleter& m_del() noexcept { return m_deleter; }
    msp_pointer_record* m_record;
    Deleter m_deleter;
  };

public:
  template<typename Ptrs, typename Spaces>
  msp_counted_deleter(Ptrs&& ptrs, Spaces&& spaces, Deleter d) noexcept 
    : m_impl(std::forward<Ptrs>(ptrs), std::forward<Spaces>(spaces), std::move(d))
  {}

  virtual void m_dispose() noexcept { 

    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      Ptr ptr = (Ptr)m_impl.m_record->m_pointers[space];
      if (ptr) {
        if (space == chai::CPU) m_impl.m_del()(ptr);
#if defined(CHAI_GPUCC)
        if (space == chai::GPU) ::chai::expt::impl::msp_dispose_on_device<<<1,1>>>(ptr, m_impl.m_del());
#endif
        SharedPtrManager::getInstance()->free(m_impl.m_record, ExecutionSpace(space));
      }
    }
  }

  virtual void m_destroy() noexcept { this->~msp_counted_deleter(); }

  virtual void moveInnerImpl() const {

    using T_non_const = std::remove_const_t<std::remove_pointer_t<Ptr>>;
    T_non_const* host_ptr = const_cast<T_non_const*>((Ptr)m_impl.m_record->m_pointers[CPU]); 

    // trigger the copy constructor
    T_non_const inner = T_non_const(*host_ptr);

    // ensure the inner type gets the state of the result of the copy
    host_ptr->operator=(inner);
  }

  msp_counted_deleter(msp_counted_deleter const&) = delete;
  msp_counted_deleter& operator=(msp_counted_deleter const&) = delete;

  msp_pointer_record* m_get_record() noexcept { return m_impl.m_record; }
private:
  impl m_impl;
};


/*
 * The interface to a given counter used by ManagedSharePtr.
 *
 * msp_record_counter will define the correct counter type based on it's
 * construction and the existance of a Deleter.
 */
class msp_record_count {
public:
  CHAI_HOST_DEVICE
  constexpr msp_record_count() noexcept : m_counted_base(0) {}

  template<typename T, typename Ptrs, typename Spaces>
  explicit msp_record_count(T, Ptrs&& ptrs, Spaces&& spaces)
  : m_counted_base( new  msp_counted_ptr<T*>(
          std::forward<Ptrs>(ptrs)
        , std::forward<Spaces>(spaces)) ) {}

  template<typename T, typename Ptrs, typename Spaces, typename Deleter>
  explicit msp_record_count(T, Ptrs&& ptrs, Spaces&& spaces, Deleter d)
  : m_counted_base( new  msp_counted_deleter<T*, Deleter>(
          std::forward<Ptrs>(ptrs)
        , std::forward<Spaces>(spaces)
        , std::move(d)) ) {}

  CHAI_HOST_DEVICE
  ~msp_record_count() noexcept
  { 
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_counted_base) {
      m_counted_base->m_release();
    }
#endif // !defined(CHAI_DEVICE_COMPILE)
  }

  CHAI_HOST_DEVICE
  msp_record_count(msp_record_count const& rhs) noexcept : m_counted_base(rhs.m_counted_base)
  { 
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_counted_base) m_counted_base->m_add_ref_copy(); 
#endif // !defined(CHAI_DEVICE_COMPILE)
  }

  CHAI_HOST_DEVICE
  msp_record_count& operator=(msp_record_count const& rhs) noexcept {
    CHAI_UNUSED_VAR(rhs);
#if !defined(CHAI_DEVICE_COMPILE)
    msp_counted_base* temp = rhs.m_counted_base;
    if (temp != m_counted_base)
    {
      if (temp) temp->m_add_ref_copy();
      if (m_counted_base) m_counted_base->m_release();
      m_counted_base = temp;
    }
#endif // !defined(CHAI_DEVICE_COMPILE)
    return *this;
  }

  void m_swap(msp_record_count& rhs) noexcept {
    msp_counted_base* temp = rhs.m_counted_base;
    rhs.m_counted_base = m_counted_base;
    m_counted_base = temp;
  }

  long m_get_use_count() const noexcept 
  { return m_counted_base ? m_counted_base->m_get_use_count() : 0; }

  friend inline bool
  operator==(msp_record_count const& a, msp_record_count const& b) noexcept
  { return a.m_counted_base == b.m_counted_base; }

  msp_pointer_record* m_get_record() const noexcept { return m_counted_base->m_get_record(); }

  template<typename Ptr>
  Ptr* m_get_pointer(chai::ExecutionSpace space) noexcept { return static_cast<Ptr*>(m_get_record()->m_pointers[space]); }

  void moveInnerImpl() const { m_counted_base->moveInnerImpl(); }

  mutable msp_counted_base* m_counted_base = nullptr;

};



}  // end of namespace expt
}  // end of namespace chai
#endif  // CHAI_SharedPointerRecord_HPP
