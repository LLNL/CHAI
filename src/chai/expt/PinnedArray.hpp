#ifndef CHAI_PINNED_ARRAY_HPP
#define CHAI_PINNED_ARRAY_HPP

#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class PinnedArray
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  template <typename T>
  class PinnedArray
    public:
      PinnedArray() = default;

      explicit PinnedArray(int allocatorID) :
        m_allocator{umpire::ResourceManager::getInstance().getAllocator(allocatorID)}
      {
      }

      explicit PinnedArray(std::size_t size, int allocatorID) :
        PinnedArray(allocatorID),
        m_size{size}
      {
        m_data = m_allocator.allocate(m_size);

        // TODO: Default initialize on host or device?
        for (std::size_t i = 0; i < size; ++i) {
          new (m_data[i]) T();
        }
      }

      PinnedArray(const PinnedArray& other) :
        m_size{other.m_size},
        m_allocator{other.m_allocator}
      {
        m_data = m_allocator.allocate(m_size);
        umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size);
      }

      PinnedArray(PinnedArray&& other) :
        m_data{other.m_data},
        m_size{other.m_size},
        m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_allocator = umpire::Allocator();
      }

      PinnedArray& operator=(const PinnedArray& other) {
        if (this != &other) { // Prevent self-assignment
          m_allocator.deallocate(m_data);
          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = m_allocator.allocate(m_size);
          umpire::ResourceManager::getInstance().copy(other.m_data, m_data, m_size);
        }

        return *this;
      }

      PinnedArray& operator=(PinnedArray&& other) {
        if (this != &other) { // Prevent self-move
          m_allocator.deallocate(m_data);
          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = other.m_data;
          
          other.m_data = nullptr;
          other.m_size = nullptr;
          other.m_allocator = umpire::Allocator();
        }

        return *this;
      }

      /*!
       * \brief Virtual destructor.
       */
      ~PinnedArray() {
        m_allocator.deallocate(m_data);
      }

      /*!
       * \brief Get the number of elements.
       */
      size_t size() const {
        return m_size;
      }

      T* data(ExecutionSpace space) {
        if (space == CPU) {
#if (__CUDACC__)
          cudaDeviceSynchronize();
#elif(__HIPCC__)
          hipDeviceSynchronize();
#endif
        }

        return m_data;
      }

    private:
      T* m_data{nullptr};
      size_t m_size{0};
      umpire::Allocator m_allocator{};
  };  // class PinnedArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_HPP
