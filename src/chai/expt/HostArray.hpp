//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_HOST_ARRAY_HPP
#define CHAI_HOST_ARRAY_HPP

namespace chai::expt
{
  template <typename T>
  class HostArray {
    public:
      HostArray() = default;

      explicit HostArray(const umpire::Allocator& allocator)
        : m_allocator{allocator}
      {
      }

      HostArray(std::size_t size, const umpire::Allocator& allocator = umpire::ResourceManager::getInstance().getAllocator("HOST"))
        : m_allocator{allocator}
      {
        resize(size);
      }

      HostArray(const HostArray& other)
        : m_allocator{other.m_allocator}
      {
        resize(other.m_size);

        if constexpr (std::is_trivially_copyable_v<T>)
        {
          std::memcpy(m_data, other.m_data, m_size * sizeof(T));
        }
        else
        {
          std::copy_n(other.m_data, m_size, m_data);
        }
      }

      HostArray(HostArray&& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_allocator{other.m_allocator}
      {
        other.m_data = nullptr;
        other.m_size = 0;
      }

      ~HostArray()
      {
        m_allocator.deallocate(m_data);
      }

      HostArray& operator=(const HostArray& other)
      {
        if (&other != this)
        {
          m_allocator.deallocate(m_data);

          m_allocator = other.m_allocator;
          m_size = other.m_size;
          m_data = static_cast<T*>(m_allocator.allocate(m_size * sizeof(T)));

          if constexpr (std::is_trivially_copyable_v<T>)
          {
            std::memcpy(m_data, other.m_data, m_size * sizeof(T));
          }
          else
          {
            std::copy_n(other.m_data, m_size, m_data);
          }
        }

        return *this;
      }

      HostArray& operator=(HostArray&& other)
      {
        if (&other != this)
        {
          m_allocator.deallocate(m_data);

          m_data = other.m_data;
          m_size = other.m_size;
          m_allocator = other.m_allocator;

          other.m_data = nullptr;
          other.m_size = 0;
        }

        return *this;
      }

      void resize(size_t newSize)
      {
        if (newSize != m_size)
        {
          T* newData = nullptr;

          if (newSize > 0)
          {
            std::size_t newSizeBytes = newSize * sizeof(T);
            newData = static_cast<T*>(m_allocator.allocate(newSizeBytes));

            if constexpr (std::is_trivially_copyable_v<T>)
            {
              std::memcpy(newData, m_data, std::min(newSizeBytes, m_size * sizeof(T)));
            }
            else
            {
              std::copy_n(m_data, std::min(newSize, m_size), newData);
            }
          }

          m_allocator.deallocate(m_data);
          m_data = newData;
          m_size = newSize;
        }
      }

      void free()
      {
        m_allocator.deallocate(m_data);
        m_data = nullptr;
        m_size = 0;
      }

      size_t size() const
      {
        return m_size;
      }

      T* data()
      {
        return m_data;
      }

      const T* data() const
      {
        return m_data;
      }

      T& operator[](std::size_t i)
      {
        return m_data[i];
      }

      const T& operator[](std::size_t i) const
      {
        return m_data[i];
      }

      T get(std::size_t i) const
      {
        return m_data[i];
      }

      void set(std::size_t i, T value)
      {
        m_data[i] = value;
      }

    private:
      T* m_data{nullptr};
      std::size_t m_size{0};
      umpire::Allocator m_allocator{umpire::ResourceManager::getInstance().getAllocator("HOST")};
  };  // class HostArray
}  // namespace chai::expt

#endif  // CHAI_HOST_ARRAY_HPP