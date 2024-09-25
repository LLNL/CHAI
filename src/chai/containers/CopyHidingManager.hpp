#ifndef CHAI_COPY_HIDING_MANAGER_HPP
#define CHAI_COPY_HIDING_MANAGER_HPP

#include <cstddef>

namespace chai {
   template <class ElementType, class Allocator = std::allocator<T>>
   class CopyHidingManager {
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

         CHAI_HOST_DEVICE CopyHidingManager(const CopyHidingManager& other) :
            m_count{other.m_count},
            m_cpu_elements{other.m_cpu_elements},
            m_gpu_elements{other.m_gpu_elements},
            m_touched{other.m_touched},
            m_allocator{other.m_allocator}
         {
            update(ArrayManager::getInstance()->getExecutionSpace());
         }

         ///
         /// Get a reference to the element at the given index
         ///
         /// @param[in]  i  Index
         ///
         /// @return a reference to the element at the given index
         ///
         CHAI_HOST_DEVICE reference operator[](size_type i) const {
#if defined(CHAI_DEVICE_COMPILE)
            return m_gpu_elements[i];
#else
            return m_cpu_elements[i];
#endif
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         CHAI_HOST_DEVICE pointer data() const {
#if !defined(CHAI_DEVICE_COMPILE)
            return data(CPU);
#else
            return data(GPU);
#endif
         }

         ///
         /// Get a pointer to the underlying data
         ///
         /// @return a pointer to the underlying data
         ///
         CHAI_HOST_DEVICE pointer data(ExecutionSpace space) const {
            update(space, true);

            if (space == CPU) {
               return m_cpu_pointer;
            }
            else if (space == GPU) {
               return m_gpu_pointer;
            }
            else {
               return nullptr;
            }
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         CHAI_HOST_DEVICE const_pointer cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
            return data(CPU);
#else
            return data(GPU);
#endif
         }

         ///
         /// Get a const pointer to the underlying data
         ///
         /// @return a const pointer to the underlying data
         ///
         CHAI_HOST_DEVICE const_pointer cdata(ExecutionSpace space) const {
            update(space, false);

            if (space == CPU) {
               return m_cpu_pointer;
            }
            else if (space == GPU) {
               return m_gpu_pointer;
            }
            else {
               return nullptr;
            }
         }

         ///
         /// Get the number of elements in the array
         ///
         /// @return the number of elements in the array
         ///
         CHAI_HOST_DEVICE size_type size() const {
            return m_count;
         }

         ///
         /// Resize the array
         ///
         /// @param[in]  count  The new number of elements
         ///
         void resize(size_type count) {
            if (count == 0) {
               free();
            }
            else if (m_count == 0) {
               // Allocate in the default space?
            }
            else {
               size_type copyCount = std::min(m_count, count);

               if (m_touched == NONE || m_touched == CPU) {
                  T* new_cpu_elements = static_cast<T*>(m_allocator.allocate(chai::CPU, count));
                  std::copy_n(m_cpu_elements, copyCount, new_cpu_elements);
                  m_allocator.deallocate(chai::CPU, m_cpu_elements);
                  m_cpu_elements = new_cpu_elements;
               }
               else if (m_touched == NONE || m_touched == GPU) {
                  T* new_gpu_elements = static_cast<T*>(m_allocator.allocate(chai::GPU, count));
                  //std::copy_n(m_cpu_elements, copyCount, new_cpu_elements);
                  m_allocator.deallocate(chai::GPU, m_gpu_elements);
                  m_gpu_elements = new_gpu_elements;
               }
            }

            m_count = count;
         }

         ///
         /// Free the array
         ///
         void free() {
            m_allocator.deallocate(chai::CPU, m_cpu_elements);
            m_cpu_elements = nullptr;

            m_allocator.deallocate(chai::GPU, m_gpu_elements);
            m_gpu_elements = nullptr;

            m_count = 0;
            m_active_elements = nullptr;
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
         size_type m_count = 0;
         T* m_cpu_elements = nullptr;
         T* m_gpu_elements = nullptr;
         chai::ExecutionSpace m_touched = chai::NONE;
         Allocator m_allocator;

         ///
         /// Update data in the given execution space
         ///
         void update(ExecutionSpace space, bool registerTouch = true) {
            if (space == CPU) {
               if (m_count > 0) {
                  if (m_cpu_elements == nullptr) {
                     m_cpu_elements = static_cast<pointer>(m_allocator.allocate(chai::CPU, m_count));
                  }

                  if (m_touched == GPU) {
                     // Copy from m_gpu_elements to m_cpu_elements
                  }

                  if (registerTouch) {
                     m_touched = CPU;
                  }
                  else {
                     m_touched = NONE;
                  }
               }
            }
            else if (space == GPU) {
               if (m_count > 0) {
                  if (m_gpu_elements == nullptr) {
                     m_gpu_elements = static_cast<pointer>(m_allocator.allocate(chai::GPU, m_count));
                  }

                  if (m_touched == CPU) {
                     // Copy from m_cpu_elements to m_gpu_elements
                  }

                  if (registerTouch) {
                     m_touched = GPU;
                  }
                  else {
                     m_touched = NONE;
                  }
               }
            }
         }
   };  // class VectorManager
}  // namespace chai

#endif  // CHAI_VECTOR_MANAGER_HPP
