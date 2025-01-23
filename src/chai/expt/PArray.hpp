#ifndef CHAI_PARRAY_HPP
#define CHAI_PARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
   template <class T>
   class PArray {
      public:
         ///
         /// Default constructor
         ///
         /// @note This is meant to be light weight, so heap allocations are
         ///       intentionally avoided (in particular, no memory manager
         ///       is created).
         ///
         PArray() = default;

         ///
         /// Construct from memory manager
         ///
         /// @param[in] manager Memory manager
         ///
         PArray(ArrayManager* manager)
             m_manager{manager}
         {
         }

         ///
         /// Copy constructor
         ///
         CHAI_HOST_DEVICE PArray(const PArray& other)
           : m_data{other.m_data},
             m_size{other.m_size},
             m_manager{other.m_manager}
         {
#if !defined(CHAI_DEVICE_COMPILE)
           if (m_manager)
           {
             m_manager->update(m_size, m_data, getCurrentExecutionSpace(), !std::is_const<T>::value);
           }
#endif
         }

         CHAI_HOST_DEVICE PArray& operator=(const PArray& other) = default;

         void free() {
           if (m_manager) {
             m_data = nullptr;
             m_size = 0;
             delete m_manager;
             m_manager = nullptr;
           }
         }

         CHAI_HOST_DEVICE size_t size() const {
            return m_size;
         }

         CHAI_HOST_DEVICE T& operator[](size_t i) const {
            return m_data[i];
         }

         T* data(ExecutionSpace space) const {
           if (m_manager) {
             return static_cast<T*>(m_manager->data(space, !std::is_const<T>::value));
           }
           else {
             return nullptr;
           }
         }

         CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
           return data(CPU);
#else
           return m_data;
#endif
         }

         const T* cdata(ExecutionSpace space) const {
           if (m_manager) {
             return static_cast<T*>(m_manager->data(space, false));
           }
           else {
             return nullptr;
           }
         }

         CHAI_HOST_DEVICE T* cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
           return cdata(CPU);
#else
           return m_data;
#endif
         }

         void update(ExecutionSpace space) const {
           if (m_manager) {
             m_manager->update(m_size, m_data, space, !std::is_const<T>::value);
           }
         }

         void update() const {
           update(CPU);
         }

         void cupdate(ExecutionSpace space) const {
           if (m_manager) {
             m_manager->update(m_size, m_data, space, false);
           }
         }

         void cupdate() const {
           cupdate(CPU);
         }

         T pick(size_t index) const {
           m_manager->pick(index);
         }

         void set(size_t index, const T& value) const {
           m_manager->set(index, value);
         }

      private:
         T* m_data = nullptr;
         size_t m_size = 0;
         ArrayManager* m_manager = nullptr;
   };  // class PArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PARRAY_HPP
