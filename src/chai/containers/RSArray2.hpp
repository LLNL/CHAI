#ifndef CHAI_RS_ARRAY_H
#define CHAI_RS_ARRAY_H

#include <cstddef>

namespace chai {
   // This is really an interface that everything else has to conform to
   template <class ElementType, class RSArrayType>
   class RSArray {
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
         /// Default constructor (defaulted)
         ///
         RSArray() = default;

         ///
         /// Default destructor (defaulted)
         ///
         ~RSArray() = default;

         ///
         /// Copy constructor (defaulted)
         ///
         RSArray(const RSArray&) = default;

         ///
         /// Copy assignment operator (defaulted)
         ///
         RSArray& operator=(const RSArray&) = default;

         ///
         /// Move constructor (deleted)
         ///
         RSArray(RSArray&&) = delete;

         ///
         /// Move assignment operator (deleted)
         ///
         RSArray& operator=(RSArray&&) = delete;

         ///
         /// Construct from an array
         ///
         /// @param[in]  array  An array
         ///
         CHAI_HOST_DEVICE explicit RSArray(const RSArrayType& array) :
            m_array{array}
         {
         }

         ///
         /// Get a reference to the element at the given index
         ///
         /// @param[in]  i  Index
         ///
         /// @return a reference to the element at the given index
         ///
         CHAI_HOST_DEVICE reference operator[](size_type i) const {
            return m_array[i];
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         CHAI_HOST_DEVICE pointer data() const {
            return m_array.data();
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         template <class ExecutionResource>
         CHAI_HOST_DEVICE pointer data(const ExecutionResource& resource) const {
            return m_array.data(resource);
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         CHAI_HOST_DEVICE const_pointer cdata() const {
            return m_array.cdata();
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         template <class ExecutionResource>
         CHAI_HOST_DEVICE const_pointer cdata(const ExecutionResource& resource) const {
            return m_array.cdata(resource);
         }

         CHAI_HOST_DEVICE void update() {
            m_array.update();
         }

         template <class ExecutionResource>
         CHAI_HOST_DEVICE void update(const ExecutionResource& resource) const {
            m_array.update(resource);
         }

         CHAI_HOST_DEVICE void cupdate() {
            m_array.cupdate();
         }

         template <class ExecutionResource>
         CHAI_HOST_DEVICE void cupdate(const ExecutionResource& resource) const {
            m_array.cupdate(resource);
         }

         CHAI_HOST_DEVICE ArrayView<ElementType, ArrayType> view() const {
            return view(0, m_count);
         }

         CHAI_HOST_DEVICE ArrayView<ElementType, ArrayType> view(size_type offset) const {
            return view(offset, m_count - offset);
         }

         CHAI_HOST_DEVICE ArrayView<ElementType, ArrayType> view(size_type offset, size_type count) const {
            return ArrayView<ElementType, ArrayType>(m_array, offset, count);
         }

         CHAI_HOST_DEVICE ArrayView<const ElementType, ArrayType> cview() const {
            return cview(0, m_count);
         }

         CHAI_HOST_DEVICE ArrayView<const ElementType, ArrayType> cview(size_type offset) const {
            return cview(offset, m_count - offset);
         }

         CHAI_HOST_DEVICE ArrayView<const ElementType, ArrayType> cview(size_type offset, size_type count) const {
            return ArrayView<const ElementType, ArrayType>(m_array, offset, count);
         }

         ///
         /// Get the number of elements in the array
         ///
         /// @return the number of elements in the array
         ///
         CHAI_HOST_DEVICE size_type size() const {
            return m_array.size();
         }

         ///
         /// Resize the array
         ///
         /// @param[in]  count  The new number of elements
         ///
         CHAI_HOST_DEVICE void resize(size_type count) {
            m_array.resize(count);
         }

         ///
         /// Free the array
         ///
         CHAI_HOST_DEVICE void clear() {
            m_array.clear();
         }

         ///
         /// Clone the array
         ///
         /// @return a clone of the array
         ///
         CHAI_HOST_DEVICE RSArray clone() {
            return RSArray(m_array.clone());
         }

      private:
         RSArrayType m_array;
   };  // class RSArray
}  // namespace chai

#endif  // CHAI_ARRAY_H
