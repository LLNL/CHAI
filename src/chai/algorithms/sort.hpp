#ifndef CHAI_SORT_HPP
#define CHAI_SORT_HPP

namespace chai {
   void sort(Array& array,
             camp::resources::Host& resource,
             const chai::Allocator& allocator) {
      ElementType* elements = array.data(resource);
      std::sort(elements, elements + array.size());
      array.update(resource);
   }

   template <class ElementType>
   void sort(RSArray<ElementType, MemoryManager>& array,
             const camp::resources::Cuda& resource,
             const chai::Allocator& allocator) {
      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      const ElementType* d_keys_in = array.data(resource);
      const size_t num_items = array.size();
      ElementType *d_keys_out = static_cast<ElementType*>(allocator.allocate(resource, num_items*sizeof(ElementType)));
      const int begin_bit = 0;
      const int end_bit = sizeof(ElementType) * 8;
      cudaStream_t stream = resource.get_stream();

      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_keys_out, num_items,
                                     begin_bit, end_bit, stream);
     
      d_temp_storage = allocator.allocate(resource, temp_storage_bytes);

      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     d_keys_in, d_keys_out, num_items,
                                     begin_bit, end_bit, stream);

      allocator.deallocate(resource, temp_storage_bytes);

      // Copy data back into original array (if allocators are not equal?)
      // Otherwise, swap?

      camp::resources::Event e = resource.get_event();
      array.set_event(e);
   }

   template <class Array>
   void sort(const Array& array,
             const camp::Resource& resource,
             const chai::Allocator& allocator) {
     
   }
}  // namespace chai

#endif  // CHAI_SORT_HPP
