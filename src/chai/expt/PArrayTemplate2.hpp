#ifndef CHAI_ARRAY_HPP
#define CHAI_ARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
   template <class T, class MemoryManager>
   class Array {
      public:
         using size_type = size_t;

         ///
         /// Default constructor
         ///
         Array() = default;

         ///
         /// Construct from a memory manager
         ///
         /// @param[in] manager Memory manager
         ///
         CHAI_HOST_DEVICE explicit Array(const MemoryManager& manager)
           : m_manager{manager}
         {
         }

         ///
         /// Construct from count and memory manager
         ///
         /// @param[in] count Number of elements
         /// @param[in] manager Memory manager
         ///
         CHAI_HOST_DEVICE explicit Array(size_type count,
                                         const MemoryManager& manager = MemoryManager())
           : Array(manager)
         {
           resize(count)
         }

         CHAI_HOST_DEVICE void resize(size_type count) {
            m_manager.resize(count);
         }

         CHAI_HOST_DEVICE void free() {
           m_manager.free();
         }

         CHAI_HOST_DEVICE size_type size() const {
            return m_manager.size();
         }

         CHAI_HOST_DEVICE T* data() const {
           return m_manager.data();
         }

         CHAI_HOST_DEVICE const T* cdata() const {
           return m_manager.cdata();
         }

         CHAI_HOST_DEVICE T& operator[](size_type i) const {
            return m_manager[i];
         }

         CHAI_HOST_DEVICE T pick(size_type i) const {
            return m_manager.pick(i);
         }

         CHAI_HOST_DEVICE void set(size_type i, T value) const {
            m_manager.set(i, value);
         }

      private:
         MemoryManager m_manager;
   };  // class Array
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_HPP
