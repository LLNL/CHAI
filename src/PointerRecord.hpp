#ifndef CHAI_PointerRecord_HPP
#define CHAI_PointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"

namespace chai {

struct PointerRecord 
{
  size_t m_size;
  void * m_pointers[NUM_EXECUTION_SPACES];
  bool m_touched[NUM_EXECUTION_SPACES];
};

}

#endif
