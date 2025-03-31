#include "chai/expt/HostManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  HostManager::HostManager(int allocatorID, std::size_t size) :
    Manager{},
    m_allocator_id{allocatorID},
    m_size{size}
  {
    m_data = umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).allocate(size);
  }

  HostManager::~HostManager() {
    umpire::ResourceManager::getInstance().getAllocator(m_allocator_id).deallocate(m_data);
  }

  std::size_t HostManager::size() const {
    return m_size;
  }

  void HostManager::update(void*& data, bool /* touch */) {
    if (execution_context() == ExecutionContext::HOST) {
      data = m_data;
    }
    else {
      data = nullptr;
    }
  }

  int HostManager::getAllocatorID() const {
    return m_allocator_id;
  }
}  // namespace expt
}  // namespace chai
