//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_DUAL_ARRAY_HPP
#define CHAI_DUAL_ARRAY_HPP

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include <cstddef>

namespace chai::expt
{
  template <typename T>
  class DualArray {
    public:
      DualArray() = default;

      DualArray(const umpire::Allocator& host_allocator,
                const umpire::Allocator& device_allocator)
        : m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
      }

      explicit DualArray(std::size_t size,
                         const umpire::Allocator& host_allocator = umpire::ResourceManager::getInstance().getAllocator("HOST"),
                         const umpire::Allocator& device_allocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE"))
        : m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
        resize(size);
      }

      DualArray(const DualArray& other)
        : m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator}
      {
        resize(other.m_size);
        umpire::ResourceManager::getInstance().copy(other.m_device_data, m_device_data, m_size * sizeof(T));
      }

      DualArray(DualArray&& other)
        : m_host_data{other.m_host_data},
          m_device_data{other.m_device_data},
          m_size{other.m_size},
          m_modified{other.m_modified},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator}
      {
        other.m_host_data = nullptr;
        other.m_device_data = nullptr;
        other.m_size = 0;
        other.m_modified = Context::NONE;
      }

      ~DualArray()
      {
        m_host_allocator.deallocate(m_host_data);
        m_device_allocator.deallocate(m_device_data);
      }

      DualArray& operator=(const DualArray& other)
      {
        if (&other != this)
        {
          m_host_allocator.deallocate(m_host_data);
          m_host_data = nullptr;

          m_device_allocator.deallocate(m_device_data);
          m_device_data = nullptr;

          m_size = 0;

          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;

          resize(other.m_size);
          // TODO: Fix the copy
          umpire::ResourceManager::getInstance().copy(other.m_device_data, m_device_data, m_size * sizeof(T));
        }

        return *this;
      }

      DualArray& operator=(DualArray&& other)
      {
        if (&other != this)
        {
          m_host_allocator.deallocate(m_host_data);
          m_device_allocator.deallocate(m_device_data);

          m_host_data = other.m_host_data;
          m_device_data = other.m_device_data;
          m_size = other.m_size;
          m_modified = other.m_modified;
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;

          other.m_host_data = nullptr;
          other.m_device_data = nullptr;
          other.m_size = 0;
          other.m_modified = Context::NONE;
        }

        return *this;
      }

      void resize(std::size_t new_size)
      {
        if (new_size != m_size)
        {
          std::size_t old_size_bytes = m_size   * sizeof(T);
          std::size_t new_size_bytes = new_size * sizeof(T);

          if (m_modified == Context::HOST ||
              (m_host_data && !m_device_data))
          {
            if (m_device_data)
            {
              m_device_allocator.deallocate(m_device_data);
              m_device_data = nullptr;
            }

            T* new_host_data = nullptr;

            if (new_size > 0)
            {
              new_host_data = static_cast<T*>(m_host_allocator.allocate(new_size_bytes));
            }

            if (m_host_data)
            {
              umpire::ResourceManager::getInstance().copy(m_host_data, new_host_data, std::min(old_size_bytes, new_size_bytes));
              m_host_allocator.deallocate(m_host_data);
            }

            m_host_data = new_host_data;
          }
          else
          {
            if (m_host_data)
            {
              m_host_allocator.deallocate(m_host_data);
              m_host_data = nullptr;
            }

            T* new_device_data = nullptr;

            if (new_size > 0)
            {
              new_device_data = static_cast<T*>(m_device_allocator.allocate(new_size_bytes));
            }

            if (m_device_data)
            {
              umpire::ResourceManager::getInstance().copy(m_device_data, new_device_data, std::min(old_size_bytes, new_size_bytes));
              m_device_allocator.deallocate(m_device_data);
            }

            m_device_data = new_device_data;
          }

          m_size = new_size;
        }
      }

      void free()
      {
        m_host_allocator.deallocate(m_host_data);
        m_host_data = nullptr;

        m_device_allocator.deallocate(m_device_data);
        m_device_data = nullptr;

        m_size = 0;
        m_modified = Context::NONE;
      }

      std::size_t size() const
      {
        return m_size;
      }

      T* data()
      {
        Context context =
          ContextManager::getInstance().getContext();

        if (context == Context::DEVICE)
        {
          if (m_device_data == nullptr)
          {
            m_device_data = static_cast<T*>(m_device_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::HOST)
          {
            umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
          }

          m_modified = Context::DEVICE;
          return m_device_data;
        }
        else if (context == Context::HOST)
        {
          if (m_host_data == nullptr)
          {
            m_host_data = static_cast<T*>(m_host_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::DEVICE)
          {
            umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
          }

          m_modified = Context::HOST;
          return m_host_data;
        }
        else
        {
          return nullptr;
        }
      }

      const T* data() const
      {
        Context context =
          ContextManager::getInstance().getContext();

        if (context == Context::DEVICE)
        {
          if (m_device_data == nullptr)
          {
            m_device_data = static_cast<T*>(m_device_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::HOST)
          {
            umpire::ResourceManager::getInstance().copy(m_host_data, m_device_data, m_size * sizeof(T));
            m_modified = Context::NONE;
          }

          return m_device_data;
        }
        else if (context == Context::HOST)
        {
          if (m_host_data == nullptr)
          {
            m_host_data = static_cast<T*>(m_host_allocator.allocate(m_size * sizeof(T)));
          }

          if (m_modified == Context::DEVICE)
          {
            umpire::ResourceManager::getInstance().copy(m_device_data, m_host_data, m_size * sizeof(T));
            m_modified = Context::NONE;
          }

          return m_host_data;
        }
        else
        {
          return nullptr;
        }
      }

      T get(std::size_t i) const
      {
        T result;

        if (m_modified == Context::DEVICE)
        {
          umpire::ResourceManager::getInstance().copy(m_device_data + i, &result, sizeof(T));
        }
        else
        {
          result = m_host_data[i];
        }

        return result;
      }

      void set(std::size_t i, T value)
      {
        if (m_modified == Context::DEVICE)
        {
          umpire::ResourceManager::getInstance().copy(&value, m_device_data + i, sizeof(T));
        }
        else
        {
          if (m_host_data == nullptr)
          {
            m_host_data = static_cast<T*>(m_host_allocator.allocate(m_size * sizeof(T)));
          }

          m_host_data[i] = value;
          m_modified = Context::HOST;
        }
      }

      const T* host_data() const
      {
        return m_host_data;
      }

      const T* device_data() const
      {
        return m_device_data;
      }

      Context modified() const
      {
        return m_modified;
      }

      umpire::Allocator host_allocator() const
      {
        return m_host_allocator;
      }

      umpire::Allocator device_allocator() const
      {
        return m_device_allocator;
      }

    private:
      mutable T* m_host_data{nullptr};
      mutable T* m_device_data{nullptr};
      std::size_t m_size{0};
      mutable Context m_modified{Context::NONE};
      mutable umpire::Allocator m_host_allocator{umpire::ResourceManager::getInstance().getAllocator("HOST")};
      mutable umpire::Allocator m_device_allocator{umpire::ResourceManager::getInstance().getAllocator("DEVICE")};
  };  // class DualArray
}  // namespace chai::expt

#endif  // CHAI_DUAL_ARRAY_HPP