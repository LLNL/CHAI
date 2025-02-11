#ifndef CHAI_MANAGED_ARRAY_HPP
#define CHAI_MANAGED_ARRAY_HPP

#include "chai/config.hpp"

namespace chai {
namespace expt {
   template <class T>
   class ManagedArray {
      public:
         ///
         /// Default constructor
         ///
         ManagedArray() = default;

         ///
         /// Construct from array manager
         ///
         /// @param[in]  manager  Array manager
         ///
         explicit ManagedArray(ArrayManager* manager = makeDefaultArrayManager())
           : m_manager{manager}
         {
         }

         ///
         /// Construct from array manager and element count
         ///
         /// @param[in]  count  Number of elements
         /// @param[in]  manager  Array manager
         ///
         explicit ManagedArray(size_t count,
                               ArrayManager* manager = makeDefaultArrayManager())
           : ManagedArray(manager)
         {
           resize(count);
         }

         ///
         /// Copy constructor
         ///
         /// This class is designed to have pointer/reference semantics,
         /// so the copy constructor performs a shallow copy.
         ///
         /// The unusual piece is that the copy constructor is also designed
         /// to trigger data copies between host and device or synchronization
         /// as needed. This enables CHAI to work in tandem with RAJA, as
         /// ManagedArrays are captured by value into lambda expressions that
         /// are passed to RAJA to execute on various backends.
         ///
         /// @param[in]  other  ManagedArray to copy from
         ///
         CHAI_HOST_DEVICE ManagedArray(const ManagedArray& other)
           : m_data{other.m_data},
             m_size{other.m_size},
             m_manager{other.m_manager}
         {
#if !defined(CHAI_DEVICE_COMPILE)
           m_manager->update(m_data, !std::is_const<T>::value, getCurrentExecutionSpace());
#endif
         }

         ///
         /// Copy assignment operator
         ///
         /// This class is designed to have pointer/reference semantics,
         /// so the copy constructor performs a shallow copy.
         ///
         /// Unlike the copy constructor, the copy assignment operator
         /// does not trigger data movement or synchronization.
         ///
         /// @param[in]  other  ManagedArray to copy from
         ///
         ManagedArray& operator=(const ManagedArray& other) = default;

         ///
         /// Resizes the ManagedArray
         ///
         /// @param[in]  count  New number of elements
         ///
         void resize(size_t count) {
           m_data = nullptr;
           m_size = count;

           if (m_manager) {
              m_manager->resize(count);
           }
           else {
              // throw
           }
         }

         ///
         /// Frees the ManagedArray
         ///
         /// @note  resize will throw after calling free
         ///
         void free() {
           m_data = nullptr;
           m_size = 0;
           delete m_manager;
           m_manager = nullptr;
         }

         ///
         /// Updates the ManagedArray to be coherent in the given space.
         /// Marks the data as touched in the given space if T is not const.
         ///
         /// If accessing a ManagedArray outside of a RAJA loop,
         /// one of the update or cupdate methods must first be called.
         ///
         /// @param[in]  space  Execution space in which to make the
         ///                    array coherent
         ///
         void update(ExecutionSpace space) const {
           if (m_manager) {
             m_manager->update(m_data, !std::is_const<T>::value, space);
           }
         }

         ///
         /// Updates the ManagedArray to be coherent on the CPU.
         /// Marks the data as touched on the CPU if T is not const.
         ///
         /// If accessing a ManagedArray outside of a RAJA loop,
         /// one of the update or cupdate methods must first be called.
         ///
         void update() const {
           update(CPU);
         }

         ///
         /// Updates the ManagedArray to be coherent in the given space.
         /// Does not mark the data as touched in the given space.
         ///
         /// If accessing a ManagedArray outside of a RAJA loop,
         /// one of the update or cupdate methods must first be called.
         ///
         /// @param[in]  space  Execution space in which to make the
         ///                    array coherent
         ///
         /// @note  Use update instead if you will modify the data.
         ///
         void cupdate(ExecutionSpace space) const {
           if (m_manager) {
             m_manager->update(m_data, false, space);
           }
         }

         ///
         /// Updates the ManagedArray to be coherent on the CPU.
         /// Does not mark the data as touched on the CPU.
         ///
         /// If accessing a ManagedArray outside of a RAJA loop,
         /// one of the update or cupdate methods must first be called.
         ///
         /// @param[in]  space  Execution space in which to make the
         ///                    array coherent
         ///
         /// @note  Use update instead if you will modify the data.
         ///
         void cupdate() const {
           cupdate(CPU);
         }

         ///
         /// Updates the ManagedArray to be coherent in the given space
         /// and returns a raw pointer that is coherent in the given space.
         /// Marks the data as touched in the given space if T is not const.
         ///
         /// @param[in]  space  Execution space in which to make the
         ///                    array coherent
         ///
         /// @return a raw pointer that is coherent in the given space
         ///
         T* data(ExecutionSpace space) const {
           update(space);
           return m_data;
         }

         ///
         /// Updates the ManagedArray to be coherent in the current space
         /// (as determined by the execution context) and returns a raw pointer
         /// that is coherent in the current space.
         /// Marks the data as touched in the current space if T is not const.
         ///
         /// @return a raw pointer that is coherent in the current space
         ///
         /// @note  If on the device, the data should already have been made
         ///        coherent and and marked as touched if appropriate.
         ///
         CHAI_HOST_DEVICE T* data() const {
#if !defined(CHAI_DEVICE_COMPILE)
           return data(CPU);
#else
           return m_data;
#endif
         }

         ///
         /// Updates the ManagedArray to be coherent in the given space
         /// and returns a raw pointer that is coherent in the given space.
         /// Does not mark the data as touched in the given space.
         ///
         /// @param[in]  space  Execution space in which to make the
         ///                    array coherent
         ///
         /// @return a raw pointer that is coherent in the given space
         ///
         const T* cdata(ExecutionSpace space) const {
           cupdate(space);
           return m_data;
         }

         ///
         /// Updates the ManagedArray to be coherent in the current space
         /// (as determined by the execution context) and returns a raw pointer
         /// that is coherent in the current space.
         /// Does not mark the data as touched in the given space.
         ///
         /// @return a raw pointer that is coherent in the current space
         ///
         /// @note  If on the device, the data should already have been made
         ///        coherent
         ///
         CHAI_HOST_DEVICE T* cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
           return cdata(CPU);
#else
           return m_data;
#endif
         }

         ///
         /// Returns the number of elements in the ManagedArray
         ///
         /// @return the number of elements in the ManagedArray
         ///
         CHAI_HOST_DEVICE size_t size() const {
            return m_size;
         }

         ///
         /// Returns the ith element in the ManagedArray
         ///
         /// @param[in]  i  Index of the element to return
         ///
         /// @return the ith element in the ManagedArray
         ///
         /// @note  Prior to calling this method, the data should have been
         ///        made coherent via the copy constructor or one of the
         ///        following methods: update, cupdate, data, or cdata.
         ///
         CHAI_HOST_DEVICE T& operator[](size_t i) const {
            return m_data[i];
         }

         ///
         /// Retrieves the ith element in the ManagedArray from the last
         /// coherent space.
         ///
         /// @param[in]  i  Index of the element to return
         ///
         /// @return the ith element in the ManagedArray
         ///
         /// @note  Pick can be an expensive operation and should be used
         ///        sparingly. If it is used repeatedly on the same
         ///        ManagedArray, one of the update/cupdate/data/cdata
         ///        methods should be considered instead.
         ///
         /// @note  One common usage is to get the last value of a ManagedArray
         ///        after an exclusive scan has been performed.
         ///
         T pick(size_t i) const {
           return m_manager->pick(index);
         }

         ///
         /// Updates the ith element in the ManagedArray in the last coherent
         /// space.
         ///
         /// @param[in]  i  Index of the element to update
         /// @param[in]  value  Sets the ith element to value
         ///
         /// @note  Set can be an expensive operation and should be used
         ///        sparingly. If it is used repeatedly on the same
         ///        ManagedArray, the update or data methods should be
         ///        considered instead.
         ///
         void set(size_t i, const T& value) const {
           m_manager->set(i, value);
         }

      private:
         T* m_data{nullptr};
         size_t m_size{0};
         ArrayManager* m_manager{getDefaultArrayManager()};
   };  // class ManagedArray
}  // namespace expt
}  // namespace chai

#endif  // CHAI_MANAGED_ARRAY_HPP
