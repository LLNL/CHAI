#ifndef CHAI_COPY_HIDING_MANAGER_HPP
#define CHAI_COPY_HIDING_MANAGER_HPP

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename HostAllocator, typename DeviceAllocator>
  class CopyHidingManager : public Manager {
    public:
      /*!
       * \brief Constructs a host array manager.
       */
      CopyHidingManager(size_t size,
                        const HostAllocator& hostAllocator = HostAllocator(),
                        const DeviceAllocator& deviceAllocator = DeviceAllocator())
        m_size{size},
        m_host_allocator{hostAllocator},
        m_device_allocator{deviceAllocator}
      {
      }

      CopyHidingManager(const CopyHidingManager&) = delete;
      CopyHidingManager& operator=(const CopyHidingManager&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~CopyHidingManager() {
        m_device_allocator.deallocate(m_device_data);
        m_host_allocator.deallocate(m_host_data);
      }

      /*!
       * \brief Get the number of elements.
       */
      virtual size_t size() const {
        return m_size;
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       *
       * \param data [out] A coherent array in the current execution space.
       */
      virtual void update(void*& data, bool touch) {
        ExecutionContext context = execution_context();

        if (context == ExecutionContext::Host) {
          if (!m_host_data) {
            m_host_data = m_host_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::Device) {
#if defined(CHAI_ENABLE_CUDA)
            cudaMemcpy(m_host_data, m_device_data, m_size, cudaMemcpyDtoH);
#elif defined(CHAI_ENABLE_HIP)
            hipMemcpy(m_host_data, m_device_data, m_size, hipMemcpyDtoH);
#else
            memcpy(m_host_data, m_device_data, m_size);
#endif

            // Reset touch
            m_touch = ExecutionContext::None;
          }

          if (touch) {
            m_touch = ExecutionContext::Host;
          }

          data = m_host_data;
        }
        else if (context == ExecutionContext::Device) {
          if (!m_device_data) {
            m_device_data = m_device_allocator.allocate(m_size);
          }

          if (m_touch == ExecutionContext::Host) {
#if defined(CHAI_ENABLE_CUDA)
            cudaMemcpy(m_device_data, m_host_data, m_size);
#elif defined(CHAI_ENABLE_HIP)
            hipMemcpy(m_device_data, m_host_data, m_size);
#else
            memcpy(m_device_data, m_host_data, m_size);
#endif

            // Reset touch
            m_touch = ExecutionContext::None;
          }

          if (touch) {
            m_touch = ExecutionContext::Device;
          }

          data = m_device_data;
        }
        else {
          data = nullptr;
        }
      }

    private:
      size_t m_size{0};
      T* m_host_data{nullptr};
      T* m_device_data{nullptr};
      HostAllocator m_host_allocator{};
      DeviceAllocator m_device_allocator{};
      ExecutionContext m_touch{ExecutionContext::None};
  };  // class CopyHidingManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_MANAGER_HPP
