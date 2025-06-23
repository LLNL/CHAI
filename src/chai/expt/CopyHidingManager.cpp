#include "chai/expt/CopyHidingManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  CopyHidingManager::CopyHidingManager(int hostAllocatorID,
                                       int deviceAllocatorID,
                                       std::size_t size) :
    Manager{},
    m_host_allocator_id{hostAllocatorID},
    m_device_allocator_id{deviceAllocatorID},
    m_size{size}
  {
  }

  CopyHidingManager::~CopyHidingManager() {
    umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).deallocate(m_device_data);
    umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).deallocate(m_host_data);
  }

  size_t CopyHidingManager::size() const {
    return m_size;
  }

  void* CopyHidingManager::data(ExecutionContext context, bool touch) {
    if (context == ExecutionContext::HOST) {
      if (!m_host_data) {
        m_host_data = umpire::ResourceManager::getInstance().getAllocator(m_host_allocator_id).allocate(m_size);
      }

      if (m_touch == ExecutionContext::DEVICE) {
        umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size);
        m_touch = ExecutionContext::NONE;
      }

      if (touch) {
        m_touch = ExecutionContext::HOST;
      }

      return m_host_data;
    }
    else if (context == ExecutionContext::DEVICE) {
      if (!m_device_data) {
        m_device_data = umpire::ResourceManager::getInstance().getAllocator(m_device_allocator_id).allocate(m_size);
      }

      if (m_touch == ExecutionContext::HOST) {
        umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size);
        m_touch = ExecutionContext::NONE;
      }

      if (touch) {
        m_touch = ExecutionContext::DEVICE;
      }

      return m_device_data;
    }
    else {
      return nullptr;
    }
  }

  int CopyHidingManager::getHostAllocatorID() const {
    return m_host_allocator_id;
  }

  int CopyHidingManager::getDeviceAllocatorID() const {
    return m_device_allocator_id;
  }

  ExecutionContext CopyHidingManager::getTouch() const {
    return m_touch;
  }
}  // namespace expt
}  // namespace chai
