//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPointerRecord_HPP
#define CHAI_SharedPointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/Types.hpp"

#include <cstddef>
#include <functional>

namespace chai
{

/*!
 * \brief Struct holding details about each pointer.
 */
//template<typename Tp>
struct msp_pointer_record {

  // Using NUM_EXECUTION_SPACES for the time being, this will help with logical
  // control since ExecutionSpaces are already defined.
  // Only CPU and GPU spaces will be used.
  // If other spaces are enabled they will not be used by ManagedSharedPtr.
  void* m_pointers[NUM_EXECUTION_SPACES];
  bool m_touched[NUM_EXECUTION_SPACES];
  bool m_owned[NUM_EXECUTION_SPACES];

  ExecutionSpace m_last_space;
  //UserCallback m_user_callback;

  int m_allocators[NUM_EXECUTION_SPACES];


  msp_pointer_record(void* host_p = nullptr, void* device_p = nullptr) : m_last_space(NONE) { 
     for (int space = 0; space < NUM_EXECUTION_SPACES; ++space ) {
        m_pointers[space] = nullptr;
        m_touched[space] = false;
        m_owned[space] = true;
        m_allocators[space] = 0;
     }
     m_pointers[CPU] = host_p;
     m_pointers[GPU] = device_p;
  }

};


class msp_counted_base {
public:
  msp_counted_base() noexcept : m_use_count(1) {}

  virtual ~msp_counted_base() noexcept {}

  virtual void m_dispose() noexcept = 0;
  virtual void m_destroy() noexcept { delete this; }

  void m_add_ref_copy() noexcept { ++m_use_count; }

  void m_release() noexcept {
    if(--m_use_count == 0) {
      m_dispose();
      m_destroy();
    }
  }

  long m_get_use_count() const noexcept { return m_use_count; }

  virtual msp_pointer_record& getPointerRecord() noexcept = 0;

private:
  msp_counted_base(msp_counted_base const&) = delete;
  msp_counted_base& operator=(msp_counted_base const&) = delete;

  long m_use_count = 0;
};

template<typename Ptr>
class msp_counted_ptr final : public msp_counted_base {
public:
  msp_counted_ptr(Ptr h_p, Ptr d_p) noexcept : m_record(h_p, d_p) {}
  virtual void m_dispose() noexcept { delete (Ptr)m_record.m_pointers[chai::CPU]; }// TODO : Other Exec spaces...
  virtual void m_destroy() noexcept { delete this; }
  msp_counted_ptr(msp_counted_ptr const&) = delete;
  msp_counted_ptr& operator=(msp_counted_ptr const&) = delete;

  msp_pointer_record& getPointerRecord() noexcept { return m_record; }
private:
  msp_pointer_record m_record;
};

template<typename Ptr, typename Deleter>
class msp_counted_deleter final : public msp_counted_base {

  class impl {
  public:
    impl(Ptr h_p, Ptr d_p, Deleter d) : m_record(h_p, d_p), m_deleter(std::move(d)) {}
    Deleter& m_del() noexcept { return m_deleter; }
    msp_pointer_record m_record;
    Deleter m_deleter;
  };

public:
  msp_counted_deleter(Ptr h_p, Ptr d_p, Deleter d) noexcept : m_impl(h_p, d_p, std::move(d)) {}
  virtual void m_dispose() noexcept { 
    printf("Delete GPU Memory Here...\n");
    m_impl.m_del()((Ptr)m_impl.m_record.m_pointers[chai::CPU]);
  }
  virtual void m_destroy() noexcept { this->~msp_counted_deleter(); }
  msp_counted_deleter(msp_counted_deleter const&) = delete;
  msp_counted_deleter& operator=(msp_counted_deleter const&) = delete;

  msp_pointer_record& getPointerRecord() noexcept { return m_impl.m_record; }
private:
  impl m_impl;
};


class msp_record_count {
public:
  constexpr msp_record_count() noexcept : m_pi(0) {}

  template<typename Ptr>
  explicit msp_record_count(Ptr h_p, Ptr d_p)
  : m_pi( new  msp_counted_ptr<Ptr>(h_p, d_p) ) {}

  template<typename Ptr, typename Deleter>
  explicit msp_record_count(Ptr h_p, Ptr d_p, Deleter d)
  : m_pi( new  msp_counted_deleter<Ptr, Deleter>(h_p, d_p, d) ) {}

  ~msp_record_count() noexcept
  { if (m_pi) m_pi->m_release(); }

  msp_record_count(msp_record_count const& rhs) noexcept : m_pi(rhs.m_pi)
  { if (m_pi) m_pi->m_add_ref_copy(); }

  msp_record_count& operator=(msp_record_count const& rhs) noexcept {
    msp_counted_base* temp = rhs.m_pi;
    if (temp != m_pi)
    {
      if (temp) temp->m_add_ref_copy();
      if (m_pi) m_pi->m_release();
      m_pi = temp;
    }
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

  msp_pointer_record& getPointerRecord() noexcept { return m_pi->getPointerRecord(); }

  template<typename Ptr>
  Ptr* getPointer(chai::ExecutionSpace space) noexcept { return static_cast<Ptr*>(getPointerRecord().m_pointers[space]); }

  msp_counted_base* m_pi;

};





}  // end of namespace chai
#endif  // CHAI_SharedPointerRecord_HPP
