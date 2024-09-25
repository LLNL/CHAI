#ifndef CHAI_MEMORY_MANAGER_H
#define CHAI_MEMORY_MANAGER_H

namespace chai {
   template <class T>
   class MemoryManager {
      public:
         virtual void resize(std::size_t count, T*& pointer) = 0;
         virtual void update(std::size_t count, T*& pointer) = 0;
   };  // class MemoryManager
}  // namespace chai

#endif  // CHAI_MEMORY_MANAGER_H
