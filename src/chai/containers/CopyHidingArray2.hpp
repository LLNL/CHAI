#ifndef CHAI_RS_COPY_HIDING_ARRAY_HPP
#define CHAI_RS_COPY_HIDING_ARRAY_HPP

#include <cstddef>

namespace chai {
   template <class ElementType>
   class CopyHidingMemoryManager {
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

         CHAI_HOST_DEVICE CopyHidingArray(const CopyHidingArray& other) :
            m_count{other.m_count},
            m_cpu_elements{other.m_cpu_elements},
            m_gpu_elements{other.m_gpu_elements},
            m_touched{other.m_touched},
            m_memory_manager{other.m_memory_manager}
         {
            update(m_memory_manager.get_current_execution_space());
         }

         CHAI_HOST_DEVICE void set_execution_space(ExecutionSpace space) {

         }

         void update(size_type count,
                     pointer& elements,
                     ExecutionSpace execution_space,
                     bool register_touch) {
            if (execution_space == NONE) {
               elements = nullptr;
            }
            else if (execution_space == CPU) {
               if (count > 0) {
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
            elements
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
