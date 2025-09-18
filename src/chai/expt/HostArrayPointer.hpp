#ifndef CHAI_HOST_ARRAY_POINTER_HPP
#define CHAI_HOST_ARRAY_POINTER_HPP

#include "chai/expt/HostArray.hpp"

namespace chai::expt
{
  template <typename T>
  class HostArrayPointer
  {
    public:
      using HostArrayType = std::conditional_t<std::is_const_v<T>, const HostArray<std::remove_cv_t<T>>, HostArray<std::remove_cv_t<T>>>;
      
      HostArrayPointer() = default;

      HostArrayPointer(HostArrayType* array)
        : m_array{array}
      {
      }

      HostArrayPointer(const HostArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_array{other.m_array}
      {
        update();
      }

      HostArrayPointer& operator=(const HostArrayPointer& other) = default;

      void update() const
      {
        if (m_array)
        {
          m_data = m_array->data();
        }
      }

      void resize(std::size_t newSize)
      {
        m_data = nullptr;
        m_size = newSize;
        m_array->resize(newSize);
      }

      void free()
      {
        m_data = nullptr;
        m_size = 0;
        delete m_array;
        m_array = nullptr;
      }

      std::size_t size() const
      {
        return m_size;
      }

      T* data() const
      {
        update();
        return m_data;
      }

      T& operator[](std::size_t i) const
      {
        return m_data[i];
      }

      T get(std::size_t i) const
      {
        return m_array->get(i);
      }

      void set(std::size_t i, T value) const
      {
        m_array->set(i, value);
      }

    private:
      T* m_data{nullptr};
      std::size_t m_size{0};
      HostArrayType* m_array{nullptr};
  };  // class HostArrayPointer
}  // namespace chai::expt

#endif  // CHAI_HOST_ARRAY_POINTER_HPP