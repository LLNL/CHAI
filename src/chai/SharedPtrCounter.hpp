
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPointerCounter_HPP
#define CHAI_SharedPointerCounter_HPP

#include <type_traits>
#include "chai/ChaiMacros.hpp"
#include "chai/SharedPtrManager.hpp"

namespace chai
{

class msp_counted_base {
public:
  msp_counted_base() noexcept : m_use_count(1) {}

  virtual ~msp_counted_base() noexcept {}

  virtual void m_dispose() noexcept = 0;
  virtual void m_destroy() noexcept { delete this; }

  virtual void moveInnerImpl() = 0;

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

template<typename Ptr>
class msp_counted_ptr final : public msp_counted_base {
public:
  msp_counted_ptr(Ptr h_p, Ptr d_p) noexcept 
    : m_record(SharedPtrManager::getInstance()->makeSharedPtrRecord(h_p, d_p, sizeof(std::remove_pointer<Ptr>), true)) 
  {}

  virtual void m_dispose() noexcept { delete (Ptr)m_record->m_pointers[chai::CPU]; }// TODO : Other Exec spaces...
  virtual void m_destroy() noexcept { delete this; }

  virtual void moveInnerImpl() {
    using T = std::remove_pointer_t<Ptr>;
    Ptr host_ptr = (Ptr) m_record->m_pointers[CPU]; 
    // trigger the copy constructor
    std::cout << "Trigger Inner Copy Ctor @ " << host_ptr << std::endl;
    T inner = T(*host_ptr);
    // ensure the inner type gets the state of the result of the copy
    host_ptr->operator=(inner);
  }

  msp_counted_ptr(msp_counted_ptr const&) = delete;
  msp_counted_ptr& operator=(msp_counted_ptr const&) = delete;
  msp_pointer_record* m_get_record() noexcept { return m_record; }
private:
  msp_pointer_record* m_record;
};

#include <typeinfo>

//template<typename T>
//void err_func(T arg){ static_assert(false); }

template<typename Ptr, typename Deleter>
class msp_counted_deleter final : public msp_counted_base {

  class impl {
  public:
    impl(Ptr h_p, Ptr d_p, Deleter d) 
      : m_record(SharedPtrManager::getInstance()->makeSharedPtrRecord(h_p, d_p, sizeof(std::remove_pointer_t<Ptr>), true))
      , m_deleter(std::move(d)) 
    {}

    Deleter& m_del() noexcept { return m_deleter; }
    msp_pointer_record* m_record;
    Deleter m_deleter;
  };

public:
  msp_counted_deleter(Ptr h_p, Ptr d_p, Deleter d) noexcept : m_impl(h_p, d_p, std::move(d)) {}
  virtual void m_dispose() noexcept { 
    printf("Delete GPU Memory Here...\n");
    m_impl.m_del()((Ptr)m_impl.m_record->m_pointers[chai::CPU]);
  }
  virtual void m_destroy() noexcept { this->~msp_counted_deleter(); }

  virtual void moveInnerImpl() {
    //using T = std::remove_cv_t<Ptr>;
    using T_non_const = std::remove_const_t<std::remove_pointer_t<Ptr>>;

    T_non_const* host_ptr = const_cast<T_non_const*>((Ptr)m_impl.m_record->m_pointers[CPU]); 
    // trigger the copy constructor
    std::cout << "Trigger Inner Copy Ctor @ " << host_ptr << std::endl;
    T_non_const inner = T_non_const(*host_ptr);

    // ensure the inner type gets the state of the result of the copy
    //err_func(host_ptr);
    host_ptr->operator=(inner);
  }

  msp_counted_deleter(msp_counted_deleter const&) = delete;
  msp_counted_deleter& operator=(msp_counted_deleter const&) = delete;

  msp_pointer_record* m_get_record() noexcept { return m_impl.m_record; }
private:
  impl m_impl;
};


class msp_record_count {
public:
  CHAI_HOST_DEVICE
  constexpr msp_record_count() noexcept : m_pi(0) {}

  template<typename Ptr>
  explicit msp_record_count(Ptr h_p, Ptr d_p)
  : m_pi( new  msp_counted_ptr<Ptr>(h_p, d_p) ) {}

  template<typename Ptr, typename Deleter>
  explicit msp_record_count(Ptr h_p, Ptr d_p, Deleter d)
  : m_pi( new  msp_counted_deleter<Ptr, Deleter>(h_p, d_p, d) ) {}

  CHAI_HOST_DEVICE
  ~msp_record_count() noexcept
  { 
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_pi) m_pi->m_release();
#endif // !defined(CHAI_DEVICE_COMPILE)
  }

  CHAI_HOST_DEVICE
  msp_record_count(msp_record_count const& rhs) noexcept : m_pi(rhs.m_pi)
  { 
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_pi) m_pi->m_add_ref_copy(); 
#endif // !defined(CHAI_DEVICE_COMPILE)
  }

  CHAI_HOST_DEVICE
  msp_record_count& operator=(msp_record_count const& rhs) noexcept {
#if !defined(CHAI_DEVICE_COMPILE)
    msp_counted_base* temp = rhs.m_pi;
    if (temp != m_pi)
    {
      if (temp) temp->m_add_ref_copy();
      if (m_pi) m_pi->m_release();
      m_pi = temp;
    }
#endif // !defined(CHAI_DEVICE_COMPILE)
    return *this;
  }

  void m_swap(msp_record_count& rhs) noexcept {
    msp_counted_base* temp = rhs.m_pi;
    rhs.m_pi = m_pi;
    m_pi = temp;
  }

  long m_get_use_count() const noexcept 
  { return m_pi ? m_pi->m_get_use_count() : 0; }

  friend inline bool
  operator==(msp_record_count const& a, msp_record_count const& b) noexcept
  { return a.m_pi == b.m_pi; }

  msp_pointer_record* m_get_record() noexcept { return m_pi->m_get_record(); }

  template<typename Ptr>
  Ptr* m_get_pointer(chai::ExecutionSpace space) noexcept { return static_cast<Ptr*>(m_get_record()->m_pointers[space]); }

  void moveInnerImpl() { m_pi->moveInnerImpl(); }

  msp_counted_base* m_pi;

};



}  // end of namespace chai
#endif  // CHAI_SharedPointerRecord_HPP
