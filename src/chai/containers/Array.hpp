#ifndef CHAI_ARRAY_H
#define CHAI_ARRAY_H

#include "MemoryManager.hpp"

namespace chai {
   template <class T>
   class Array {
      public:
         Array() = default;

         Array(MemoryManager<T>* manager) :
            m_manager{manager}
         {
         }

         Array(std::size_t count, MemoryManager<T>* manager) :
            m_count{count},
            m_manager{manager}
         {
            m_manager->resize(m_count, m_data);
         }

         Array(const Array& other) :
            m_count{other.m_count},
            m_data{other.m_data},
            m_manager{other.m_manager}
         {
#if !defined(CHAI_DEVICE_COMPILE)
            m_manager->update(m_count, m_data);
#endif
         }

         CHAI_HOST_DEVICE std::size_t size() const { return m_count; }

         template <typename Index>
         CHAI_HOST_DEVICE T& operator[](Index i) const { return m_data[i]; }

         CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
            m_manager->update(m_count, m_data);
#endif
            return m_data;
         }

      private:
         std::size_t m_count = 0;
         T* m_data = nullptr;
         MemoryManager<T>* m_manager = nullptr;
   };  // class Array
}  // namespace chai

#endif  // CHAI_ARRAY_H
