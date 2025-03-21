#ifndef CHAI_PINNED_MANAGER_HPP
#define CHAI_PINNED_MANAGER_HPP

namespace chai {
namespace expt {
  /*!
   * \class PinnedManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename Allocator>
  class PinnedManager : public Manager {
    public:
      /*!
       * \brief Constructs a host array manager.
       */
      PinnedManager(size_t size,
                    const Allocator& allocator = Allocator())
        m_size{size},
        m_data{allocator.allocate(size)},
        m_allocator{allocator}
      {
      }

      PinnedManager(const PinnedManager&) = delete;
      PinnedManager& operator=(const PinnedManager&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~PinnedManager() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       *
       * \param data [out] A coherent array in the current execution space.
       */
      virtual void update(void*& data, bool touch) {
        ExecutionContext context = execution_context();

        if (context == ExecutionContext::None) {
          data = nullptr;
        }
        else {
           if (context == ExecutionContext::Host) {
             // TODO: Only sync if last touched on device
             synchronizeDeviceIfNeeded();
           }

           data = m_data;
        }
      }

    private:
      size_t m_size{0};
      T* m_data{nullptr};
      Allocator m_allocator{};
  };  // class PinnedManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_MANAGER_HPP
