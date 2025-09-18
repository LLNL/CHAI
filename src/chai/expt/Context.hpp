#ifndef CHAI_CONTEXT_HPP
#define CHAI_CONTEXT_HPP

namespace chai::expt
{
  enum class Context
  {
    NONE = 0,
    HOST = 1,
    DEVICE = 2
  };
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_HPP