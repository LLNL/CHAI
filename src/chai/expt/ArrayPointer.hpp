#ifndef CHAI_ARRAY_POINTER_HPP
#define CHAI_ARRAY_POINTER_HPP

#include "chai/config.hpp"

namespace chai::expt
{
  template <typename T, template <typename> ArrayType>
  class ArrayPointer
  {
    public:
      using Array = std::conditional_t<std::is_const_v<T>, const <std::remove_cv_t<T>>, ArrayType<std::remove_cv_t<T>>>;
      
      ArrayPointer() = default;

      explicit ArrayPointer(Array* array)
        : m_array{array}
      {
      }

      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_array{other.m_array}
      {
        update();
      }

      template <typename OtherT, 
                std::enable_if_t<std::is_convertible_v<OtherT (*)[], T (*)[]>* = nullptr>
      CHAI_HOST_DEVICE Array(const Array<OtherElementType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_array{other.m_array}
      {
      }

      ArrayPointer& operator=(const ArrayPointer& other) = default;

      void resize(std::size_t newSize)
      {
        if (m_array == nullptr)
        {
          m_array = new Array();
        }

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

      CHAI_HOST_DEVICE std::size_t size() const
      {
        return m_size;
      }

      CHAI_HOST_DEVICE void update() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_array)
        {
          m_data = m_array->data();
        }
#endif
      }

      CHAI_HOST_DEVICE T* data() const
      {
        update();
        return m_data;
      }

      CHAI_HOST_DEVICE T& operator[](std::size_t i) const
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
      Array* m_array{nullptr};
  };  // class ArrayPointer
}  // namespace chai::expt

#endif  // CHAI_ARRAY_POINTER_HPP