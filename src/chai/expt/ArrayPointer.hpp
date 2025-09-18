#ifndef CHAI_ARRAY_POINTER_HPP
#define CHAI_ARRAY_POINTER_HPP

#include "chai/config.hpp"
#include <type_traits>

namespace chai::expt
{
  template <typename ElementType, template <typename> typename ArrayType>
  class ArrayPointer
  {
    public:
      using Array = std::conditional_t<std::is_const_v<ElementType>, const <std::remove_cv_t<ElementType>>, ArrayType<std::remove_cv_t<ElementType>>>;
      
      ArrayPointer() = default;

      CHAI_HOST_DEVICE ArrayPointer(std::nullptr_t)
        : ArrayPointer()
      {
      }

      explicit ArrayPointer(Array* array)
        : m_array{array}
      {
        update();
      }

      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_array{other.m_array}
      {
        update();
      }

      template <typename OtherT, 
                std::enable_if_t<std::is_convertible_v<OtherT (*)[], ElementType (*)[]>* = nullptr>
      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer<OtherT>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_array{other.m_array}
      {
        update();
      }

      CHAI_HOST_DEVICE ArrayPointer& operator=(const ArrayPointer& other)
      {
        if (&other != this)
        {
          m_data = other.m_data;
          m_size = other.m_size;
          m_array = other.m_array;

          update();
        }

        return *this;
      }

      CHAI_HOST_DEVICE ArrayPointer& operator=(std::nullptr_t)
      {
        m_data = nullptr;
        m_size = 0;
        m_array = nullptr;

        return *this;
      }

      void resize(std::size_t newSize)
      {
        if (m_array == nullptr)
        {
          m_array = new Array();
        }

        m_data = nullptr;
        m_size = newSize;
        m_array->resize(newSize);

        update();
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
          if (ElementType* data = m_array->data(); data)
          {
            m_data = data;
          }

          m_size = m_array->size();
        }
#endif
      }

      CHAI_HOST_DEVICE void cupdate() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_array)
        {
          const Array* array = m_array;

          if (ElementType* data = array->data(); data)
          {
            m_data = data;
          }

          m_size = array->size();
        }
#endif
      }

      CHAI_HOST_DEVICE ElementType* data() const
      {
        update();
        return m_data;
      }

      CHAI_HOST_DEVICE ElementType* cdata() const
      {
        cupdate();
        return m_data;
      }

      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i) const
      {
        return m_data[i];
      }

      ElementType get(std::size_t i) const
      {
        if (m_array && i < m_array->size())
        {
          return m_array->get(i);
        }
        else
        {
          throw std::out_of_range("Array index out of bounds");
        }
      }

      void set(std::size_t i, ElementType value) const
      {
        if (m_array && i < m_array->size())
        {
          m_array->set(i, value);
        }
        else
        {
          throw std::out_of_range("Array index out of bounds");
        }
      }

    private:
      ElementType* m_data{nullptr};
      std::size_t m_size{0};
      Array* m_array{nullptr};
  };  // class ArrayPointer
}  // namespace chai::expt

#endif  // CHAI_ARRAY_POINTER_HPP