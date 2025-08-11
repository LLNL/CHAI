#include "chai/expt/CopyHidingManager.hpp"
#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  CopyHidingManager::CopyHidingManager(const umpire::Allocator& cpuAllocator,
                                       const umpire::Allocator& gpuAllocator)
    : Manager{},
      m_cpu_allocator{cpuAllocator},
      m_gpu_allocator{gpuAllocator}
  {
  }

  CopyHidingManager::CopyHidingManager(int cpuAllocatorID,
                                       int gpuAllocatorID)
    : Manager{},
      m_resource_manager{umpire::ResourceManager::getInstance()},
      m_cpu_allocator{m_resource_manager.getAllocator(cpuAllocatorID)},
      m_gpu_allocator{m_resource_manager.getAllocator(gpuAllocatorID)}
  {
  }

  CopyHidingManager::CopyHidingManager(size_type size) :
    : Manager{},
      m_size{size}
  {
    // TODO: Exception handling
    m_cpu_data = m_cpu_allocator.allocate(size);
    m_gpu_data = m_gpu_allocator.allocate(size);
  }

  CopyHidingManager::CopyHidingManager(size_type size,
                                       const umpire::Allocator& cpuAllocator,
                                       const umpire::Allocator& gpuAllocator) :
    : Manager{},
      m_cpu_allocator{cpuAllocator},
      m_gpu_allocator{gpuAllocator},
      m_size{size}
  {
    // TODO: Exception handling
    m_cpu_data = m_cpu_allocator.allocate(size);
    m_gpu_data = m_gpu_allocator.allocate(size);
  }

  CopyHidingManager::CopyHidingManager(size_type size,
                                       int cpuAllocatorID,
                                       int gpuAllocatorID) :
    : Manager{},
      m_resource_manager{umpire::ResourceManager::getInstance()},
      m_cpu_allocator{m_resource_manager.getAllocator(cpuAllocatorID)},
      m_gpu_allocator{m_resource_manager.getAllocator(gpuAllocatorID)},
      m_size{size}
  {
    // TODO: Exception handling
    m_cpu_data = m_cpu_allocator.allocate(size);
    m_gpu_data = m_gpu_allocator.allocate(size);
  }

  CopyHidingManager::CopyHidingManager(const CopyHidingManager& other)
    : Manager{},
      m_cpu_allocator{other.m_cpu_allocator},
      m_gpu_allocator{other.m_gpu_allocator},
      m_size{other.m_size},
      m_touch{other.m_touch}
  {
    if (other.m_cpu_data)
    {
      m_cpu_data = m_cpu_allocator.allocate(m_size);
      m_resourceManager.copy(m_cpu_data, other.m_cpu_data, m_size);
    }

    if (other.m_gpu_data)
    {
      m_gpu_data = m_gpu_allocator.allocate(m_size);
      m_resourceManager.copy(m_gpu_data, other.m_gpu_data, m_size);
    }
  }

  CopyHidingManager::CopyHidingManager(CopyHidingManager&& other) noexcept
    : Manager{},
      m_cpu_allocator{other.m_cpu_allocator},
      m_gpu_allocator{other.m_gpu_allocator},
      m_size{other.m_size},
      m_touch{other.m_touch},
      m_cpu_data{other.m_cpu_data},
      m_gpu_data{other.m_gpu_data}
  {
    other.m_size = 0;
    other.m_cpu_data = nullptr;
    other.m_gpu_data = nullptr;
    other.m_touch = ExecutionContext::NONE;
  }

  CopyHidingManager::~CopyHidingManager() {
    m_cpu_allocator.deallocate(m_cpu_data);
    m_gpu_allocator.deallocate(m_gpu_data);
  }

  CopyHidingManager& CopyHidingManager::operator=(const CopyHidingManager& other)
  {
    if (this != &other)
    {
      // Copy-assign base class if needed (uncomment if Manager is copy-assignable)
      // Manager::operator=(other);

      // Copy-assign or copy members
      m_cpu_allocator = other.m_cpu_allocator;
      m_gpu_allocator = other.m_gpu_allocator;
      m_touch = other.m_touch;

      // Allocate new resources before releasing old ones for strong exception safety
      void* new_cpu_data = nullptr;
      void* new_gpu_data = nullptr;

      if (other.m_cpu_data)
      {
        new_cpu_data = m_cpu_allocator.allocate(other.m_size);
        m_resourceManager.copy(new_cpu_data, other.m_cpu_data, other.m_size);
      }

      if (other.m_gpu_data)
      {
        new_gpu_data = m_gpu_allocator.allocate(other.m_size);
        m_resourceManager.copy(new_gpu_data, other.m_gpu_data, other.m_size);
      }

      // Clean up old resources
      if (m_cpu_data)
      {
        m_cpu_allocator.deallocate(m_cpu_data, m_size);
      }

      if (m_gpu_data)
      {
        m_gpu_allocator.deallocate(m_gpu_data, m_size);
      }

      // Assign new resources and size
      m_cpu_data = new_cpu_data;
      m_gpu_data = new_gpu_data;
      m_size = other.m_size;
    }

    return *this;
  }

  CopyHidingManager& CopyHidingManager::operator=(CopyHidingManager&& other) noexcept
  {
    if (this != &other)
    {
      // Release any resources currently held
      if (m_cpu_data)
      {
        m_cpu_allocator.deallocate(m_cpu_data, m_size);
        m_cpu_data = nullptr;
      }
      if (m_gpu_data)
      {
        m_gpu_allocator.deallocate(m_gpu_data, m_size);
        m_gpu_data = nullptr;
      }

      // Move-assign base class if needed (uncomment if Manager is move-assignable)
      // Manager::operator=(std::move(other));

      // Move-assign or copy members
      m_cpu_allocator = other.m_cpu_allocator;
      m_gpu_allocator = other.m_gpu_allocator;
      m_size = other.m_size;
      m_cpu_data = other.m_cpu_data;
      m_gpu_data = other.m_gpu_data;
      m_touch = other.m_touch;

      // Null out other's pointers and reset size
      other.m_cpu_data = nullptr;
      other.m_gpu_data = nullptr;
      other.m_size = 0;
      other.m_touch = ExecutionContext::NONE;
    }
    return *this;
  }

  void CopyHidingManager::resize(size_type newSize)
  {
    if (newSize != m_size)
    {
      if (m_touch == ExecutionContext::CPU)
      {
        m_resource_manager.reallocate(m_cpu_pointer, newSize);

        if (m_gpu_pointer)
        {
          m_resource_manager.deallocate(m_gpu_pointer);
          m_gpu_pointer = m_gpu_allocator.allocate(newSize);
        }
      }
      else if (m_touch == ExecutionContext::GPU)
      {
        m_resource_manager.reallocate(m_gpu_pointer, newSize);

        if (m_cpu_pointer)
        {
          m_resource_manager.deallocate(m_cpu_pointer);
          m_cpu_pointer = m_cpu_allocator.allocate(newSize);
        }
      }
      else
      {
        if (m_gpu_pointer)
        {
          m_resource_manager.reallocate(m_gpu_pointer, newSize);
        }

        if (m_cpu_pointer)
        {
          m_resource_manager.reallocate(m_cpu_pointer, newSize);
        }
      }
    }
  }

  size_type CopyHidingManager::size() const
  {
    return m_size;
  }

  void* CopyHidingManager::data(bool touch)
  {
    ExecutionContext context = ExecutionContextManager::getCurrentContext();

    if (context == ExecutionContext::CPU)
    {
      if (!m_cpu_data)
      {
        m_cpu_data = m_cpu_allocator.allocate(m_size);
      }

      if (m_touch == ExecutionContext::GPU)
      {
        m_resource_manager.copy(m_cpu_data, m_gpu_data, m_size);
        m_touch = ExecutionContext::NONE;
      }

      if (touch)
      {
        m_touch = ExecutionContext::CPU;
      }

      return m_cpu_data;
    }
    else if (context == ExecutionContext::GPU)
    {
      if (!m_gpu_data)
      {
        m_gpu_data = m_gpu_allocator.allocate(m_size);
      }

      if (m_touch == ExecutionContext::CPU)
      {
        m_resource_manager.copy(m_gpu_data, m_cpu_data, m_size);
        m_touch = ExecutionContext::NONE;
      }

      if (touch)
      {
        m_touch = ExecutionContext::GPU;
      }

      return m_gpu_data;
    }
    else
    {
      return nullptr;
    }
  }
}  // namespace expt
}  // namespace chai
