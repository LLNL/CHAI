//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPointerRecord_HPP
#define CHAI_SharedPointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/SharedPtrManager.hpp"
#include "chai/Types.hpp"

#include <cstddef>
#include <functional>

namespace chai
{
namespace expt
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
  // TODO: Iplement user callbacks for ManagedSharedPtr.
  //UserCallback m_user_callback;

  int m_allocators[NUM_EXECUTION_SPACES];

  msp_pointer_record() :
    m_last_space(CPU) { 
    for (int space = 0; space < NUM_EXECUTION_SPACES; ++space ) {
      m_pointers[space] = nullptr;
      m_touched[space] = false;
      m_owned[space] = true;
      m_allocators[space] = 0;
    }
  }

};





}  // end of namespace expt
}  // end of namespace chai
#endif  // CHAI_SharedPointerRecord_HPP
