#ifndef CHAI_COPY_HIDING_MEMORY_MANAGER_HPP
#define CHAI_COPY_HIDING_MEMORY_MANAGER_HPP

namespace chai {
   template <class T>
   class CopyHidingMemoryManager {
      public:
         CopyHidingMemoryManager() = default;

         void resize(std::size_t& count, T*& pointer, std::size_t newCount) {
            
         }

         void update(std::size_t count, T*& pointer) {

         }

      private:
         umpire::Allocator m_hostAllocator = umpire::ResourceManager::getInstance().getAllocator("HOST");
         umpire::Allocator m_deviceAllocator = umpire::ResourceManager::getInstance().getAllocator("DEVICE");
   };  // class MemoryManager
}  // namespace chai

#endif  // CHAI_COPY_HIDING_MEMORY_MANAGER_HPP
