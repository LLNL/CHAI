#ifndef CHAI_PINNED_ARRAY_HPP
#define CHAI_PINNED_ARRAY_HPP

namespace chai {
namespace expt {
  /*!
   * \class PinnedArray
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename Allocator>
  class PinnedArray : public Manager {
    public:
      PinnedArray() noexcept(noexcept(Allocator())) :
        PinnedArray(Allocator())
      {
      }

      explicit PinnedArray(const Allocator& allocator) :
        m_allocator{allocator}
      {
      }

      /*!
       * \brief Constructs a host array manager.
       */
      PinnedArray(size_t size,
                    const Allocator& allocator = Allocator()) :
        m_size{size},
        m_data{allocator.allocate(size)},
        m_allocator{allocator}
      {
      }

      PinnedArray(const PinnedArray& other) :
        m_size{other.m_size},
        m_data{other.m_allocator.allocate(size)},
        m_allocator{other.m_allocator}
      {
        // Copy data from other array
      }

      PinnedArray(PinnedArray&& other) :
        m_size{other.m_size},
        m_data{other.m_data},
        m_allocator{other.m_allocator}
      {
        other.m_size = 0;
        other.m_data = nullptr;
        other.m_allocator = Allocator();
      }

      PinnedArray& operator=(const PinnedArray&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~PinnedArray() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Get the number of elements.
       */
      virtual size_t size() const {
        return m_size;
      }

      virtual T* data(ExecutionContext context, bool touch) {

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
  };  // class PinnedArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_HPP
