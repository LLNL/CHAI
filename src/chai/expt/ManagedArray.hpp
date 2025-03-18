#ifndef CHAI_MANAGED_ARRAY_HPP
#define CHAI_MANAGED_ARRAY_HPP

namespace chai {
  template <typename T>
  class ManagedArray {
    public:
      ManagedArray() = default;

      ManagedArray(ArrayManager* manager) :
        m_manager{manager}
      {
        if (m_manager) {
          m_manager->update(m_data, m_size);
        }
      }

      void resize(size_t count) {
        if (count == 0) {
          free();
        }
        else {
          if (!m_manager) {
            makeDefaultArrayManager(count);
          }
          else {
            m_manager->resize(count);
          }

          m_manager->update(m_data, m_size);
        }
      }

      void free() {
        m_data = nullptr;
        m_size = 0;
        delete m_manager;
        m_manager = nullptr;
      }

      CHAI_HOST_DEVICE T& operator[](size_t i) const {
        return m_data[i];
      }

      CHAI_HOST_DEVICE size_t size() const {
        return m_size;
      }

    private:
      T* m_data = nullptr;
      size_t m_size = 0;
      ArrayManager* m_manager = nullptr;
  };  // class ManagedArray

  template <typename Manager, typename... Args>
  ManagedArray<T> makeManagedArray(Args&&... args) {
    return ManagedArray<T>(new Manager(std::forward<Args>(args)...));
  }
}  // namespace chai

#endif  // CHAI_MANAGED_ARRAY_HPP
