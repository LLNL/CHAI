#include "chai/expt/Manager.hpp"

namespace chai {
namespace expt {
  ExecutionContext& Manager::execution_space() {
    static thread_local ExecutionContext s_space = ExecutionContext::NONE;
    return s_space;
  }
}  // namespace expt
}  // namespace chai
