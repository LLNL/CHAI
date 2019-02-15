#ifndef MANAGED_PTR_H_
#define MANAGED_PTR_H_

#include "chai/ChaiMacros.hpp"
#include "chai/ManagedArray.hpp"

#include "../util/forall.hpp"

// Standard libary headers
#include <cstddef>

namespace chai {
   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   /// This wrapper calls new on both the GPU and CPU so that polymorphism can
   ///    be used on the GPU. It is modeled after std::shared_ptr, so it does
   ///    reference counting and automatically cleans up when the last reference
   ///    is destroyed. If we ever do multi-threading on the CPU, locking will
   ///    need to be added to the reference counter.
   /// Requirements:
   ///    The actual type created (D in the first constructor) must be copy
   ///       constructible and the copy constructor must call the base class
   ///       copy constructors to avoid slicing. The default constructor has
   ///       the correct behavior, but only if the whole inheritance chain uses
   ///       the default copy constructor. Otherwise, at whatever point the
   ///       default is no longer used, the the base class copy constructors
   ///       must be called to avoid copy slicing. If the copy constructors are
   ///       private, this class must be declared as a friend.
   ///    The actual type created (D in the first constructor) must be convertible
   ///       to T (e.g. T is a base class of D).
   ///    This wrapper does NOT automatically sync the GPU copy if the CPU copy is
   ///       updated and vice versa. The one exception to this is that if the class
   ///       has chai::ManagedArray members or members that inherit chai::ManagedArray,
   ///       these will be kept in sync (hence the need for D to be copy constructible).
   ///       HOWEVER, the chai::ManagedArray members must be initialized upon object
   ///       construction. Otherwise, if you wish to keep the CPU and GPU copies in sync,
   ///       you must explicitly call the same modifying function in the CPU context and
   ///       in the GPU context.
   ///    Do NOT pass the base pointer type to the constructor. Always pass the derived
   ///       type.
   ///    Pointer types members of T must be chai::ManagedArrays or managed_ptrs to be
   ///       accessible on the GPU. The same requirements apply to inner managed_ptrs.
   ///       Requirements for using chai::ManagedArrays in the proper context still apply.
   ///    Methods that can be called on the CPU and GPU must be declared with the
   ///       __host__ __device__ specifiers. This includes the constructors (including
   ///       copy constructors) and destructors.
   ///    Raw pointer members still can be used, but they will only be valid on the host.
   ///       To prevent accidentally using them in a device context, any methods that
   ///       access raw pointers should be host only.
   ///    Be especially careful of passing raw pointers to member functions.
   template <typename T>
   class managed_ptr {
      public:
         using element_type = T;

         ///
         /// @author Alan Dayton
         ///
         /// Default constructor.
         /// Initializes the reference count to 0.
         ///
         CHAI_HOST constexpr managed_ptr() noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Construct from nullptr.
         /// Initializes the reference count to 0.
         ///
         CHAI_HOST constexpr managed_ptr(std::nullptr_t) noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given pointer.
         /// Takes the given host pointer and creates a copy of the object on the GPU.
         ///
         /// @param[in] cpuPtr The host pointer to take ownership of
         ///
         template <typename D>
         CHAI_HOST explicit managed_ptr(D* ptr) :
            m_cpu(ptr),
            m_numReferences(new std::size_t{1}) {
#ifdef __CUDACC__
            createDevicePtr(*ptr);
#endif

            // Need to be able to access the copy constructor if T is a base pointer type
            m_copyConstructor = [] (void* basePtr) {
               D derivedCopy = *(static_cast<D*>(basePtr));
               (void) derivedCopy;
            };
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr& other) noexcept :
            m_cpu(other.m_cpu),
#ifdef __CUDACC__
            m_gpu(other.m_gpu),
#endif
            m_numReferences(other.m_numReferences),
            m_copyConstructor(other.m_copyConstructor),
            m_destructor(other.m_destructor) {

#ifndef __CUDA_ARCH__
               // Increment the number of references.
               (*m_numReferences)++;

               // Trigger copy constructor so that any ManagedArrays in the object
               // are copied to the right data space.
               m_copyConstructor(m_cpu);
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///    D must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         template <typename D>
         CHAI_HOST_DEVICE managed_ptr(managed_ptr<D> const & other) noexcept :
            m_cpu(other.m_cpu),
#ifdef __CUDACC__
            m_gpu(other.m_gpu),
#endif
            m_numReferences(other.m_numReferences),
            m_copyConstructor(other.m_copyConstructor),
            m_destructor(other.m_destructor) {
#ifndef __CUDA_ARCH__
               // Increment the number of references.
               (*m_numReferences)++;

               // Trigger copy constructor so that any ManagedArrays in the object
               // are copied to the right data space.
               m_copyConstructor(m_cpu);
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Destructor. Decreases the reference count and if this is the last reference,
         ///    clean up.
         ///
         CHAI_HOST_DEVICE ~managed_ptr() {
#ifndef __CUDA_ARCH__
            if (m_numReferences) {
               (*m_numReferences)--;

               if (m_numReferences && *m_numReferences == 0) {
                  delete m_numReferences;

                  m_destructor(m_cpu);

#ifdef __CUDACC__
                  destroyDevicePtr();
#endif
               }
            }
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy assignment operator.
         /// Copies the given managed_ptr and increases the reference count.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr& other) noexcept {
            if (this != &other) {
               m_cpu = other.m_cpu;
#ifdef __CUDACC__
               m_gpu = other.m_gpu;
#endif
               m_numReferences = other.m_numReferences;
               m_copyConstructor = other.m_copyConstructor;
               m_destructor = other.m_destructor;

#ifndef __CUDA_ARCH__
               (*m_numReferences)++;

               // Trigger copy constructor so that any ManagedArrays in the object
               // are copied to the right data space.
               m_copyConstructor(m_cpu);
#endif
            }

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Conversion copy assignment operator.
         /// Copies the given managed_ptr and increases the reference count.
         ///    D must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         template<class D>
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr<D>& other) noexcept {
            if (this != &other) {
               m_cpu = other.m_cpu;
#ifdef __CUDACC__
               m_gpu = other.m_gpu;
#endif
               m_numReferences = other.m_numReferences;
               m_copyConstructor = other.m_copyConstructor;
               m_destructor = other.m_destructor;

#ifndef __CUDA_ARCH__
               (*m_numReferences)++;

               // Trigger copy constructor so that any ManagedArrays in the object
               // are copied to the right data space.
               m_copyConstructor(m_cpu);
#endif
            }

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* get() const {
#ifndef __CUDA_ARCH__
            return m_cpu;
#else
            return m_gpu;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* operator->() const {
#ifndef __CUDA_ARCH__
            return m_cpu;
#else
            return m_gpu;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU reference depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T& operator*() const {
#ifndef __CUDA_ARCH__
            return *m_cpu;
#else
            return *m_gpu;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the number of managed_ptrs owning these pointers.
         ///
         CHAI_HOST std::size_t use_count() const {
            if (m_numReferences) {
               return *m_numReferences;
            }
            else {
               return 0;
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns true if the contained pointer is not nullptr, false otherwise.
         ///
         CHAI_HOST_DEVICE inline explicit operator bool() const noexcept {
#ifndef __CUDA_ARCH__
            return m_cpu != nullptr;
#else
            return m_gpu != nullptr;
#endif
         }

#ifdef __CUDACC__
         ///
         /// @author Alan Dayton
         ///
         /// Creates the device pointer.
         /// Should be called only by the constructor, but extended __host__ __device__
         ///    lambdas can only be in public methods.
         ///
         template <typename D>
         CHAI_HOST void createDevicePtr(const D& obj) {
            chai::ManagedArray<D*> temp(1, chai::GPU);

            forall(cuda(), 0, 1, [=] __device__ (int i) {
               temp[i] = new D(obj);
            });

            temp.move(chai::CPU);
            m_gpu = temp[0];
            temp.free();

            // __host__ __device__ functions can't be created in constructor
            // Need to be able to delete the original type if T is a base pointer type
            m_destructor = [] CHAI_HOST_DEVICE (void* basePtr) {
               delete static_cast<D*>(basePtr);
            };
         }

         ///
         /// @author Alan Dayton
         ///
         /// Cleans up the device pointer.
         /// Should be called only by the destructor, but extended __host__ __device__
         ///    lambdas can only be in public methods.
         ///
         CHAI_HOST void destroyDevicePtr() {
            chai::ManagedArray<T*> temp(1, chai::CPU);
            temp[0] = m_gpu;

            // Does not capture "this"
            void (*destructor)(void*) = m_destructor;

            forall(cuda(), 0, 1, [=] __device__ (int i) {
               destructor(temp[i]);
            });

            temp.free();
         }
#endif

      private:
         T* m_cpu = nullptr; /// The host pointer

#ifdef __CUDACC__
         T* m_gpu = nullptr; /// The device pointer
#endif

         size_t* m_numReferences = nullptr; /// The reference counter

         void (*m_copyConstructor)(void*); /// A function that casts to the derived type and calls the copy constructor so that chai::ManagedArrays are moved to the correct execution space.

         void (*m_destructor)(void*); /// A function that casts to the derived type and calls delete on it.

         template <class D> friend class managed_ptr; /// Needed for the converting constructor
   };

   /// Comparison operators

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to compare
   /// @param[in] rhs The second managed_ptr to compare
   ///
   template <class T, class U>
   inline bool operator==(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
      return lhs.get() == rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to compare
   /// @param[in] rhs The second managed_ptr to compare
   ///
   template <class T, class U>
   inline bool operator!=(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
      return lhs.get() != rhs.get();
   }

   /// Comparison operators with nullptr

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] lhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   inline bool operator==(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.get() == nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] rhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   inline bool operator==(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
      return nullptr == rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] lhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   inline bool operator!=(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.get() != nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] rhs The managed_ptr to compare to nullptr
   ///
   template<class T>
   inline bool operator!=(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
      return nullptr != rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @params[in] args The arguments to T's constructor
   ///
   template <typename T, typename... Args>
   managed_ptr<T> make_managed(Args&&... args) {
      return managed_ptr<T>(new T(std::forward<Args>(args)...));
   }
} // namespace chai

#endif // MANAGED_PTR

