#ifndef CHAI_PointerRecord_HPP
#define CHAI_PointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"

namespace chai {

/*
 * \brief Struct holding details about each pointer.
 */
struct PointerRecord 
{
  size_t m_size;
  void * m_pointers[NUM_EXECUTION_SPACES];
  bool m_touched[NUM_EXECUTION_SPACES];
};

} // end of namespace chai

#endif // CHAI_PointerRecord_HPP
