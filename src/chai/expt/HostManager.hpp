#ifndef CHAI_HOST_MANAGER_HPP
#define CHAI_HOST_MANAGER_HPP

namespace chai {
namespace expt {
  /*!
   * \class HostManager
   *
   * \brief Controls the coherence of an array on the CPU.
   */
  template <typename Allocator>
  class HostManager : public Manager {
    public:
      /*!
       * \brief Constructs a host array manager.
       */
      HostManager(size_t size, const Allocator& allocator) :
        m_allocator{allocator},
        m_size{size}
      {
        m_data = m_allocator.allocate(size);
      }

      HostManager(const HostManager&) = delete;
      HostManager& operator=(const HostManager&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~HostManager() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       *
       * \param data [out] A coherent array in the current execution space.
       */
      virtual void update(void*& data, bool touch) {
        if (execution_space() == ExecutionSpace::CPU) {
          data = m_data;
        }
        else {
          data = nullptr;
        }
      }

    private:
      Allocator m_allocator{};
      size_t m_size{0};
      T* m_data{nullptr};
  };  // class HostManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_HOST_MANAGER_HPP
