#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

namespace chai {

/*!
 * Enum listing possible execution spaces.
 */
enum ExecutionSpace { 
  NONE = 0, /**< Default, no execution space. */
  CPU, /**< Executing in CPU space */
  GPU, /**< Execution in GPU space */
  // NUM_EXECUTION_SPACES should always be last!
  NUM_EXECUTION_SPACES /**< Used to count total number of spaces */
};

} // end of namespace chai

#endif // CHAI_ExecutionSpaces_HPP
