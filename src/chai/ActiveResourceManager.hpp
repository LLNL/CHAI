//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ActiveResourceManager_HPP
#define CHAI_ActiveResourceManager_HPP

#include "camp/resource.hpp"

#include <array>
#include <vector>

namespace chai
{

/*!
 * \Class to store list of Resource pointers. Holds data on the stack 
 * until a certain threshold, then uses heap memory.
 */
class ActiveResourceManager {

  /*!
   * Size of array on the stack.
   */
  static constexpr int BASE_SIZE = 16;

  /*!
   * Base array on the stack.
   */
  std::array<camp::resources::Resource*, BASE_SIZE> m_res_base;

  /*!
   * Heap containter for extra resources if more than BASE_SIZE pushed.
   */
  std::vector<camp::resources::Resource*> m_res_overflow;

  /*!
   * Current number of active resources in the list.
   */
  int m_size = 0;

public:
  ActiveResourceManager();

  /*!
   * Retrun current size of the resource list. 
   */
  int size();

  /*!
   * Push a new resource onto the list. 
   */
  void push_back(camp::resources::Resource* res);

  /*!
   * Clear all values on the heap and set m_size to 0. 
   */
  void clear();

  /*!
   * Check if empty.
   */
  bool is_empty() const;

  /*!
   * Get resource at given index. 
   */
  camp::resources::Resource* operator [](int i) const;
};

}  // end of namespace chai

#include "chai/ActiveResourceManager.inl"

#endif  // CHAI_ActiveResourceManager_HPP
