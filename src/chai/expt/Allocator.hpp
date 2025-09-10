#ifndef CHAI_ALLOCATOR_HPP
#define CHAI_ALLOCATOR_HPP

namespace chai::expt {
  class Allocator {
    private:
      class AllocatorConcept
      {
        public:
          virtual ~AllocatorConcept() = default;
          virtual void* do_allocate(std::size_t bytes) = 0;
          virtual void do_deallocate(void* ptr) = 0;
          virtual std::unique_ptr<AllocatorConcept> clone() const = 0;
      };  // class AllocatorConcept

      template <typename AllocatorType>
      class AllocatorModel : public AllocatorConcept
      {
        public:
          AllocatorModel(AllocatorType allocator)
            : m_allocator{std::move(allocator)}
          {
          }

          virtual void* allocate(std::size_t bytes) override
          {
            return allocate(m_allocator, bytes);
          }

          virtual void do_deallocate(void* ptr) override
          {
            deallocate(m_allocator, ptr);
          }

          virtual std::unique_ptr<AllocatorConcept> clone() const override
          {
            return std::make_unique<AllocatorModel>(*this);
          }

        private:
          AllocatorType m_allocator;
      };  // class AllocatorModel

      friend void* allocate(const Allocator& allocator, std::size_t bytes)
      {
        return allocator.m_pimpl->do_allocate(bytes);
      }

      friend void deallocate(const Allocator& allocator, void* ptr)
      {
        allocator.m_pimple->do_deallocate(ptr);
      }

      std::unique_ptr<AllocatorConcept> m_pimpl;

    public:
      template <typename AllocatorType>
      Allocator(AllocatorType allocator)
        : m_pimpl{std::make_unique<AllocatorModel<AllocatorType>>(std::move(allocator))}
      {
      }

      Allocator(const Allocator& other)
        : m_pimple{other.m_pimpl->clone()}
      { 
      }

      Allocator& operator=(const Allocator& other)
      {
        Allocator temp(other);
        std::swap(m_pimpl, temp.m_pimpl);
        return *this;
      }
  };  // class Allocator
}

#endif  // CHAI_ALLOCATOR_HPP