#ifndef CHAI_VECTOR_MANAGER_HPP
#define CHAI_VECTOR_MANAGER_HPP

#include <cstddef>
#include <vector>

namespace chai {
   template <class ElementType, class Allocator = std::allocator<T>>
   class VectorManager {
      public:
         ///
         /// Member types
         ///
         using size_type = std::size_t;
         using value_type = ElementType;
         using reference = ElementType&;
         using const_reference = const ElementType&;
         using pointer = ElementType*;
         using const_pointer = const ElementType*;

         ///
         /// Get a reference to the element at the given index
         ///
         /// @param[in]  i  Index
         ///
         /// @return a reference to the element at the given index
         ///
         reference operator[](size_type i) const {
            return (*m_elements)[i];
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         pointer data() const {
            return m_elements->data();
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         const_pointer cdata() const {
            return m_elements->data();
         }

         ///
         /// Get the number of elements in the array
         ///
         /// @return the number of elements in the array
         ///
         size_type size() const {
            return m_elements->size();
         }

         ///
         /// Resize the array
         ///
         /// @param[in]  count  The new number of elements
         ///
         void resize(size_type count) {
            if (m_elements == nullptr) {
               m_elements = new std::vector<T, Allocator>(count);
            }
            else {
               m_elements->resize(count);
            }
         }

         ///
         /// Free the array
         ///
         void free() {
            delete m_elements;
            m_elements = nullptr;
         }

         ///
         /// Clone the array
         ///
         /// @return a clone of the array
         ///
         VectorManager clone() {
            VectorManager manager;

            if (m_elements != nullptr) {
               manager.m_elements = new std::vector<T, Allocator>(*m_elements);
            }

            return manager;
         }

      private:
         std::vector<T, Allocator>* m_elements = nullptr;
   };  // class VectorManager
}  // namespace chai

#endif  // CHAI_VECTOR_MANAGER_HPP
