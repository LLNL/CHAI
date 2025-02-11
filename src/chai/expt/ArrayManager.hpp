#ifndef CHAI_ARRAY_MANAGER_HPP
#define CHAI_ARRAY_MANAGER_HPP

namespace chai {
namespace expt {
   class ArrayManager {
      public:
         virtual void allocate(size_t count) = 0;
         virtual void reallocate(size_t count) = 0;
         virtual void deallocate(size_t count) = 0;

         void resize(size_t count) {
           if (count > 0) {

           }
            
         }

         void free();

         void update(T*& data, bool touch, ExecutionSpace space);

         void update(T*& data, bool touch);

         T pick(size_t i) const;

         void set(size_t i, const T& value) const;



      private:
         MemoryModel* m_model{nullptr};


   };  // class ArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_MANAGER_HPP
