#ifndef CHAI_EXECUTION_CONTEXT_HPP
#define CHAI_EXECUTION_CONTEXT_HPP

namespace chai {
namespace expt {
  /*!
   * \brief Enum listing possible execution contexts.
   */
  enum class ExecutionContext {
    /*! Default, no execution space. */
    NONE = 0,
    /*! Executing in CPU space */
    CPU,
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
    /*! Executing in GPU space */
    GPU
#endif
  };  // enum class ExecutionContext
}  // namespace expt
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_HPP
