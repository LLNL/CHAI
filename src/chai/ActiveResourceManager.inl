//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ActiveResourceManager_INL
#define CHAI_ActiveResourceManager_INL

#include "ActiveResourceManager.hpp"

namespace chai
{

CHAI_INLINE
int ActiveResourceManager::size() {
  return m_size;
}


CHAI_INLINE
void ActiveResourceManager::push_back(camp::resources::Resource * res) {
  if (m_size < BASE_SIZE) {
    m_res_base[m_size] = res;
  }
  else {
    m_res_overflow.push_back(res); 
  }

  m_size++;
}


CHAI_INLINE
void ActiveResourceManager::clear() {
  m_res_overflow.clear();
  m_size = 0;
}


CHAI_INLINE
bool ActiveResourceManager::is_empty() const {
  return m_size == 0;
}


CHAI_INLINE
camp::resources::Resource* ActiveResourceManager::operator[](int i) const {
  if (i < 0 || i >= m_size) {
    return nullptr;
  }
  else {
    return i < BASE_SIZE ? m_res_base[i] : m_res_overflow[i - BASE_SIZE];
  }
}

} //end of namespace chai

#endif // CHAI_ActiveResourceManager_INL
