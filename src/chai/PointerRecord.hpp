//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_PointerRecord_HPP
#define CHAI_PointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/Types.hpp"

#include <cstddef>
#include <functional>

namespace chai
{

/*!
 * \brief Struct holding details about each pointer.
 */
struct PointerRecord {
  /*!
   * Size of pointer allocation in bytes
   */
  std::size_t m_size;

  /*!
   * Array holding the pointer in each execution space.
   */
  void* m_pointers[NUM_EXECUTION_SPACES];

  /*!
   * Array holding touched state of pointer in each execution space.
   */
  bool m_touched[NUM_EXECUTION_SPACES];

  /*!
   * Execution space where this arary was last touched.
   */
  ExecutionSpace m_last_space;

  /*!
   * Array holding ownership status of each pointer.
   */
  bool m_owned[NUM_EXECUTION_SPACES];


  /*!
   * User defined callback triggered on memory operations.
   *
   * Function is passed the execution space that the memory is
   * moved to, and the number of bytes moved.
   */
  UserCallback m_user_callback;

  int m_allocators[NUM_EXECUTION_SPACES];

  /*!
   * \brief Default constructor
   *
   */
  PointerRecord() : m_size(0), m_last_space(NONE) { 
     m_user_callback = [] (const PointerRecord*, Action, ExecutionSpace) {};
     for (int space = 0; space < NUM_EXECUTION_SPACES; ++space ) {
        m_pointers[space] = nullptr;
        m_touched[space] = false;
        m_owned[space] = true;
        m_allocators[space] = 0;
     }
  }
};

}  // end of namespace chai

#endif  // CHAI_PointerRecord_HPP
