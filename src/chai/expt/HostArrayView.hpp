#ifndef CHAI_HOST_ARRAY_VIEW_HPP
#define CHAI_HOST_ARRAY_VIEW_HPP

namespace chai::expt
{

template <typename T>
class HostArrayView
{
  public:
    using HostArrayType = std::conditional_t<std::is_const_v<T>, const HostArray<std::remove_cv_t<T>>, HostArray<std::remove_cv_t<T>>>;
    
    HostArrayView() = default;

    HostArrayView(HostArrayType& array)
      : m_array{std::addressof(array)}
    {
    }

    HostArrayView(const HostArrayView& other)
      : m_data{other.m_data},
        m_size{other.m_size},
        m_array{other.m_array}
    {
      update();
    }

    HostArrayView& operator=(const HostArrayView& other) = default;

    void update() const
    {
      if (m_array)
      {
        m_data = m_array->data();
      }
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
};  // class HostArrayView

}  // namespace chai::expt

#endif  // CHAI_HOST_ARRAY_VIEW_HPP