#ifndef CHAI_RS_ARRAY_VIEW_HPP
#define CHAI_RS_ARRAY_VIEW_HPP

#include <cstddef>

namespace chai {
   // This is really an interface that everything else has to conform to
   template <class ElementType, class MemoryManager>
   class RSArrayView {
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
         /// Default constructor
         ///
         RSArrayView() = default;

         ///
         /// Construct from a memory manager
         ///
         /// @param[in]  memory_manager  Memory manager
         ///
         CHAI_HOST_DEVICE explicit RSArrayView(const RSArray<ElementType, MemoryManager>& rsArray) :
            m_count{rsArray.size()},
            m_memory_manager{rsArray.get_memory_manager()}
         {
            m_memory_manager.update(m_count, m_elements);
         }

         ///
         /// Copy constructor
         ///
         CHAI_HOST_DEVICE RSArrayView(const RSArrayView& other) :
            m_count{other.m_count},
            m_elements{other.m_elements},
            m_memory_manager{other.m_memory_manager}
         {
            m_memory_manager.update(m_count,
                                    m_elements,
                                    get_current_resource(),
                                    !std::is_const<ElementType>::value);
         }

         ///
         /// Get a copy of the memory manager
         ///
         /// @return a copy of the memory manager
         ///
         CHAI_HOST_DEVICE MemoryManager get_memory_manager() const {
            return m_memory_manager;
         }

         ///
         /// Get a reference to the element at the given index
         ///
         /// @param[in]  i  Index
         ///
         /// @return a reference to the element at the given index
         ///
         CHAI_HOST_DEVICE reference operator[](size_type i) const {
            return m_elements[i];
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         CHAI_HOST_DEVICE pointer data() const {
#if !defined(CHAI_DEVICE_COMPILE)
            m_memory_manager.set_execution_space(chai::CPU);
#else
            m_memory_manager.set_execution_space(chai::GPU);
#endif
            m_memory_manager.update(m_count, m_elements);
            return m_elements;
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         CHAI_HOST_DEVICE const_pointer cdata() const {
            return m_memory_manager.update(m_count, m_elements, resource, false);
         }

         CHAI_HOST_DEVICE void update(resource, event) {
            m_memory_manager.update(m_count, m_elements, resource

         }

         ///
         /// Get the number of elements in the array
         ///
         /// @return the number of elements in the array
         ///
         CHAI_HOST_DEVICE size_type size() const {
            return m_count;
         }

      private:
         size_type m_count = 0;
         T* m_elements = nullptr;
         MemoryManager m_memory_manager;
   };  // class RSArray
}  // namespace chai

#endif  // CHAI_RS_ARRAY_VIEW_HPP
