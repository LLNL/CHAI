#ifndef CHAI_PARRAY_HPP
#define CHAI_PARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
   template <class T, class MemoryManager>
   class PArray {
      public:
         using size_type = size_t;

         ///
         /// Default constructor
         ///
         CHAI_HOST_DEVICE PArray() noexcept(noexcept(MemoryManager()))
           : PArray(MemoryManager())
         {
         }

         ///
         /// Construct from a memory manager
         ///
         /// @param[in] manager Memory manager
         ///
         explicit PArray(const MemoryManager& manager)
           : m_manager{manager}
         {
         }

         ///
         /// Construct from count and memory manager
         ///
         /// @param[in] count Number of elements
         /// @param[in] manager Memory manager
         ///
         explicit PArray(size_type count,
                         const MemoryManager& manager = MemoryManager())
           : m_data{manager.allocate(count)},
             m_size{count},
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
           m_manager.update(m_size, m_data, !std::is_const<T>::value);
#endif
         }

         CHAI_HOST_DEVICE PArray& operator=(const PArray& other) = default;

         void resize(size_t size) {
            if (size == 0) {
               free();
            }
            else if (m_manager) {

            }
            else {
               throw;
            }
         }

         void free() {
           if (m_manager) {
             delete m_manager;
             m_manager = nullptr;
             m_data = nullptr;
             m_size = 0;
           }
         }

         CHAI_HOST_DEVICE size_t size() const {
            return m_size;
         }

         CHAI_HOST_DEVICE T& operator[](size_t i) const {
            return m_data[i];
         }

         T* data(ExecutionLocation location,
                 bool touch = !std::is_const<T>::value) const {
           if (m_manager) {
             m_data = static_cast<T*>(m_manager->get(location, touch));
           }

           return m_data;
         }

         CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
           return data(ExecutionLocation::Host);
#else
           return m_data;
#endif
         }

      private:
         T* m_data = nullptr;
         size_t m_size = 0;
         MemoryManager* m_manager = nullptr;
   };  // class PArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_PARRAY_HPP
