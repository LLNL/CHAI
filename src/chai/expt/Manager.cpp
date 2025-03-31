#include "chai/expt/Manager.hpp"

namespace chai {
namespace expt {
  ExecutionContext& Manager::execution_context() {
    static thread_local ExecutionContext s_context = ExecutionContext::NONE;
    return s_context;
  }
}  // namespace expt
}  // namespace chai
