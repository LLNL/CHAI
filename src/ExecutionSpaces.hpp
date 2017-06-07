#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

#include "chai/config.hpp"

namespace chai {

/*!
 * \brief Enum listing possible execution spaces.
 */
enum ExecutionSpace { 
  /*! Default, no execution space. */
  NONE = 0,
  /*! Executing in CPU space */
  CPU,
#if defined(ENABLE_CUDA)
  /*! Execution in GPU space */
  GPU,
#endif
  // NUM_EXECUTION_SPACES should always be last!
  /*! Used to count total number of spaces */
  NUM_EXECUTION_SPACES
};

} // end of namespace chai

#endif // CHAI_ExecutionSpaces_HPP
