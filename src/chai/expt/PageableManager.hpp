#ifndef CHAI_PAGEABLE_MANAGER_HPP
#define CHAI_PAGEABLE_MANAGER_HPP

#include "chai/expt/PinnedManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \alias PageableManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename Allocator>
  using PageableManager = PinnedManager<Allocator>;
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PAGEABLE_MANAGER_HPP
