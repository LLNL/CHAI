#ifndef CHAI_MALLOC_MEMORY_MANAGER_HPP
#define CHAI_MALLOC_MEMORY_MANAGER_HPP

namespace chai {
   template <class T>
   class MallocMemoryManager {
      public:
         void resize(std::size_t count) {
            if (count == 0) {
               std::free(m_pointer);
               m_pointer = nullptr;
            }
            else {
               m_pointer = static_cast<T*>(std::realloc(m_pointer, count));
            }

            m_count = count;
         }

         std::size_t size() const {
            return m_count;
         }

         T* data() const {
            return m_pointer;
         }

      private:
         std::size_t m_count = 0;
         T* m_pointer = nullptr;
   };  // class MallocMemoryManager
}  // namespace chai

#endif  // CHAI_MALLOC_MEMORY_MANAGER_HPP
