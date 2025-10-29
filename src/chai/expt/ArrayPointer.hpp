#ifndef CHAI_ARRAY_POINTER_HPP
#define CHAI_ARRAY_POINTER_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include <type_traits>

namespace chai::expt
{
  template <typename ElementType, template <typename> typename ManagerType>
  class ArrayPointer
  {
    public:
      using Manager = ManagerType<std::remove_cv_t<ElementType>>;
      
      ArrayPointer() = default;

      CHAI_HOST_DEVICE ArrayPointer(std::nullptr_t)
        : ArrayPointer()
      {
      }

      explicit ArrayPointer(Manager* array)
        : m_manager{array}
      {
        update();
      }

      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }

#if 0
      template <typename OtherElementType, 
                typename = std::enable_if_t<std::is_convertible_v<OtherElementType (*)[], ElementType (*)[]>>>
      CHAI_HOST_DEVICE ArrayPointer(const ArrayPointer<OtherElementType, ManagerType>& other)
        : m_data{other.m_data},
          m_size{other.m_size},
          m_manager{other.m_manager}
      {
        update();
      }
#endif

      CHAI_HOST_DEVICE ArrayPointer& operator=(const ArrayPointer& other)
      {
        if (&other != this)
        {
          m_data = other.m_data;
          m_size = other.m_size;
          m_manager = other.m_manager;

          update();
        }

        return *this;
      }

      CHAI_HOST_DEVICE ArrayPointer& operator=(std::nullptr_t)
      {
        m_data = nullptr;
        m_size = 0;
        m_manager = nullptr;

        return *this;
      }

      void resize(std::size_t newSize)
      {
        if (m_manager == nullptr)
        {
          m_manager = new Manager();
        }

        m_data = nullptr;
        m_size = newSize;
        m_manager->resize(newSize);

        update();
      }

      void free()
      {
        m_data = nullptr;
        m_size = 0;
        delete m_manager;
        m_manager = nullptr;
      }

      CHAI_HOST_DEVICE std::size_t size() const
      {
        return m_size;
      }

      CHAI_HOST_DEVICE void update()
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager)
        {
          if (ElementType* data = m_manager->data(); data)
          {
            m_data = data;
          }

          m_size = m_manager->size();
        }
#endif
      }

      CHAI_HOST_DEVICE void cupdate()
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager)
        {
          if (ElementType* data = m_manager->data(); data)
          {
            m_data = data;
          }

          m_size = m_manager->size();
        }
#endif
      }

      CHAI_HOST_DEVICE ElementType* data()
      {
        update();
        return m_data;
      }

      CHAI_HOST_DEVICE ElementType* cdata()
      {
        cupdate();
        return m_data;
      }

      CHAI_HOST_DEVICE ElementType& operator[](std::size_t i)
      {
        return m_data[i];
      }

      ElementType get(std::size_t i)
      {
        if (m_manager && i < m_manager->size())
        {
          return m_manager->get(i);
        }
        else
        {
          throw std::out_of_range("Manager index out of bounds");
        }
      }

      void set(std::size_t i, ElementType value)
      {
        if (m_manager && i < m_manager->size())
        {
          m_manager->set(i, value);
        }
        else
        {
          throw std::out_of_range("Manager index out of bounds");
        }
      }

    private:
      ElementType* m_data{nullptr};
      std::size_t m_size{0};
      Manager* m_manager{nullptr};
  };  // class ArrayPointer
}  // namespace chai::expt

#endif  // CHAI_ARRAY_POINTER_HPP