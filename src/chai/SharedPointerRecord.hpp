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
template<typename Tp>
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

  //template<typename Yp>
  //msp_pointer_record(Yp* host_p = nullptr, Yp* device_p = nullptr) : m_last_space(NONE) { 
  //   for (int space = 0; space < NUM_EXECUTION_SPACES; ++space ) {
  //      m_pointers[space] = nullptr;
  //      m_touched[space] = false;
  //      m_owned[space] = true;
  //      m_allocators[space] = 0;
  //   }
  //   m_pointers[CPU] = host_p;
  //   m_pointers[GPU] = device_p;
  //}


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

  //Tp* get_pointer(ExecutionSpace space) noexcept { return m_pointers[space]; }
  //template<typename Yp>
  //msp_pointer_record(msp_pointer_record<Yp> const& rhs) :
  //  m_pointers(rhs.m_pointers),
  //  m_touched(rhs.m_touched),
  //  m_owned(rhs.m_owned),
  //  m_last_space(rhs.m_last_space),
  //  m_allocators(rhs.m_allocators)
  //{}



};




}  // end of namespace chai

#endif  // CHAI_SharedPointerRecord_HPP
