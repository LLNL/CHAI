#ifndef CHAI_PointerRecord_HPP
#define CHAI_PointerRecord_HPP

#include "chai/ExecutionSpaces.hpp"

namespace chai {

/*!
 * \brief Struct holding details about each pointer.
 */
struct PointerRecord 
{
  /*!
   * Size of pointer allocation in bytes
   */
  size_t m_size;

  /*!
   * Array holding the pointer in each execution space.
   */
  void * m_pointers[NUM_EXECUTION_SPACES];

  /*!
   * Array holding touched state of pointer in each execution space.
   */
  bool m_touched[NUM_EXECUTION_SPACES];
};

} // end of namespace chai

#endif // CHAI_PointerRecord_HPP
