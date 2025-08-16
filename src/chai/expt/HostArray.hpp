#ifndef CHAI_HOST_ARRAY_HPP
#define CHAI_HOST_ARRAY_HPP

#include "umpire/ResourceManager.hpp"
#include <cstddef>  // for size_t

namespace chai {
namespace expt {
  /*!
   * \class HostArray
   *
   * \brief Manages an array in host memory with RAII semantics.
   * 
   * This class provides a simpler alternative to CopyHidingArray
   * when only host memory access is needed.
   */
  template <typename ElementT>
  class HostArray {
    public:
      using size_type = std::size_t;

      /*!
       * \brief Default constructor creates an empty array.
       */
      HostArray() = default;

      /*!
       * \brief Constructs a HostArray with the given Umpire allocator.
       */
      HostArray(const umpire::Allocator& allocator) :
        m_allocator{allocator}
      { 
      }

      /*!
       * \brief Constructs a HostArray with the given Umpire allocator ID.
       */
      HostArray(int allocatorID) :
        m_resource_manager{umpire::ResourceManager::getInstance()},
        m_allocator{m_resource_manager.getAllocator(allocatorID)}
      {
      }

      /*!
       * \brief Constructs a HostArray with the given size using the default allocator.
       */
      HostArray(size_type size) :
        m_size{size}
      {
        if (size > 0) {
          m_data = static_cast<ElementT*>(m_allocator.allocate(size * sizeof(ElementT)));
        }
      }

      /*!
       * \brief Constructs a HostArray with the given size using the specified allocator.
       */
      HostArray(size_type size, const umpire::Allocator& allocator) :
        m_allocator{allocator},
        m_size{size}
      {
        if (size > 0) {
          m_data = static_cast<ElementT*>(m_allocator.allocate(size * sizeof(ElementT)));
        }
      }

      /*!
       * \brief Constructs a HostArray with the given size using the specified allocator ID.
       */
      HostArray(size_type size, int allocatorID) :
        m_resource_manager{umpire::ResourceManager::getInstance()},
        m_allocator{m_resource_manager.getAllocator(allocatorID)},
        m_size{size}
      {
        if (size > 0) {
          m_data = static_cast<ElementT*>(m_allocator.allocate(size * sizeof(ElementT)));
        }
      }

      /*!
       * \brief Constructs a deep copy of the given HostArray.
       */
      HostArray(const HostArray& other) :
        m_allocator{other.m_allocator},
        m_size{other.m_size}
      {
        if (m_size > 0) {
          m_data = static_cast<ElementT*>(m_allocator.allocate(m_size * sizeof(ElementT)));

          for (size_type i = 0; i < m_size; ++i) {
            m_data[i] = other.m_data[i];
          }
        }
      }

      /*!
       * \brief Constructs a HostArray that takes ownership of the resources from the given HostArray.
       */
      HostArray(HostArray&& other) noexcept :
        m_allocator{other.m_allocator},
        m_size{other.m_size},
        m_data{other.m_data}
      {
        other.m_size = 0;
        other.m_data = nullptr;
      }

      /*!
       * \brief Destructor releases allocated memory.
       */
      ~HostArray()
      {
        if (m_data) {
          m_allocator.deallocate(m_data);
        }
      }

      /*!
       * \brief Copy assignment operator.
       */
      HostArray& operator=(const HostArray& other)
      {
        if (this != &other) {
          // Allocate new data before releasing old data
          ElementT* new_data = nullptr;
          
          if (other.m_size > 0) {
            new_data = static_cast<ElementT*>(other.m_allocator.allocate(other.m_size * sizeof(ElementT)));
            
            for (size_type i = 0; i < other.m_size; ++i) {
              new_data[i] = other.m_data[i];
            }
          }
          
          // Clean up old data
          if (m_data) {
            m_allocator.deallocate(m_data);
          }
          
          // Update allocator, size, and data
          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = new_data;
        }
        
        return *this;
      }

      /*!
       * \brief Move assignment operator.
       */
      HostArray& operator=(HostArray&& other) noexcept
      {
        if (this != &other) {
          // Clean up current resources
          if (m_data) {
            m_allocator.deallocate(m_data);
          }
          
          // Move resources from other
          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = other.m_data;
          
          // Reset other
          other.m_size = 0;
          other.m_data = nullptr;
        }
        
        return *this;
      }

      /*!
       * \brief Resize the array.
       * 
       * If the new size is larger, the existing content is preserved and
       * new elements are default-initialized. If the new size is smaller,
       * only the first newSize elements are preserved.
       */
      void resize(size_type newSize)
      {
        if (newSize != m_size) {
          ElementT* new_data = nullptr;
          
          if (newSize > 0) {
            new_data = static_cast<ElementT*>(m_allocator.allocate(newSize * sizeof(ElementT)));
            
            // Copy existing data, up to the smaller of m_size and newSize
            size_type copy_size = (newSize < m_size) ? newSize : m_size;
            for (size_type i = 0; i < copy_size; ++i) {
              new_data[i] = m_data[i];
            }
            
            // Initialize new elements if expanding
            for (size_type i = m_size; i < newSize; ++i) {
              new_data[i] = ElementT();
            }
          }
          
          // Clean up old data
          if (m_data) {
            m_allocator.deallocate(m_data);
          }
          
          m_data = new_data;
          m_size = newSize;
        }
      }

      /*!
       * \brief Get the size of the array.
       */
      size_type size() const
      {
        return m_size;
      }

      /*!
       * \brief Get access to the underlying data.
       */
      ElementT* data()
      {
        return m_data;
      }

      /*!
       * \brief Get const access to the underlying data.
       */
      const ElementT* data() const
      {
        return m_data;
      }

      /*!
       * \brief Array subscript operator for element access.
       */
      ElementT& operator[](size_type index)
      {
        return m_data[index];
      }

      /*!
       * \brief Const array subscript operator for element access.
       */
      const ElementT& operator[](size_type index) const
      {
        return m_data[index];
      }

      /*!
       * \brief Check if the array is empty.
       */
      bool empty() const
      {
        return m_size == 0;
      }

      /*!
       * \brief Clear the array by deallocating memory and setting size to 0.
       */
      void clear()
      {
        if (m_data) {
          m_allocator.deallocate(m_data);
          m_data = nullptr;
        }
        m_size = 0;
      }

    private:
      umpire::ResourceManager& m_resource_manager{umpire::ResourceManager::getInstance()};
      umpire::Allocator m_allocator{m_resource_manager.getAllocator("HOST")};
      size_type m_size{0};
      ElementT* m_data{nullptr};
  };  // class HostArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_HOST_ARRAY_HPP