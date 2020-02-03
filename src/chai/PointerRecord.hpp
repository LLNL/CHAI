//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_PointerRecord_HPP
#define CHAI_PointerRecord_HPP

#include "chai/ActiveResourceManager.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/Types.hpp"

#include "camp/contexts.hpp"

#include <cstddef>
#include <functional>
#include <array>

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


  bool transfer_pending;
  camp::resources::Event m_event;
  camp::resources::Context* m_last_context = nullptr;
  ActiveResourceManager m_res_manager;
};

}  // end of namespace chai

#endif  // CHAI_PointerRecord_HPP
