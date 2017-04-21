#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

namespace chai {

enum ExecutionSpace { 
  NONE = 0,
  CPU,
  GPU,
  /* 
   * NUM_EXECUTION_SPACES should always be last!
   */
  NUM_EXECUTION_SPACES };

} // end of namespace chai

#endif // CHAI_ExecutionSpaces_HPP
