#ifndef MANAGED_PTR_H_
#define MANAGED_PTR_H_

#include "chai/ChaiMacros.hpp"
#include "chai/ManagedArray.hpp"

#include "../util/forall.hpp"

// Standard libary headers
#include <cstddef>
#include <tuple>

namespace chai {

#ifdef __CUDACC__
   namespace detail {
      ///
      /// @author Alan Dayton
      ///
      /// Creates a new T on the device.
      ///
      /// @param[out] devicePtr Used to return the device pointer to the new T
      /// @param[in]  args The arguments to T's constructor
      ///
      /// @note Cannot capture argument packs in an extended device lambda,
      ///       so explicit kernel is needed.
      ///
      template <typename T, typename... Args>
      __global__ void createDevicePtr(T*& devicePtr, Args... args)
      {
         devicePtr = new T(std::forward<Args>(args)...);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Destroys the device pointer.
      ///
      /// @param[out] devicePtr The device pointer to call delete on
      ///
      template <typename T>
      __global__ void destroyDevicePtr(T*& devicePtr)
      {
         if (devicePtr) {
            delete devicePtr;
         }
      }

      ///
      /// @author Alan Dayton
      ///
      /// Creates a new T on the device.
      ///
      /// @param[in]  args The arguments to T's constructor
      ///
      /// @return The device pointer to the new T
      ///
      template <typename T, typename... Args>
      CHAI_HOST T* createDevicePtr(Args&&... args) {
         T* devicePtr;
         createDevicePtr<<<1, 1>>>(devicePtr, args...);
         cudaDeviceSynchronize();
         return devicePtr;
      }
   }
#endif

   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   /// This wrapper calls new on both the GPU and CPU so that polymorphism can
   ///    be used on the GPU. It is modeled after std::shared_ptr, so it does
   ///    reference counting and automatically cleans up when the last reference
   ///    is destroyed. If we ever do multi-threading on the CPU, locking will
   ///    need to be added to the reference counter.
   /// Requirements:
   ///    The actual type created (U in the first constructor) must be convertible
   ///       to T (e.g. T is a base class of U or there is a user defined conversion).
   ///    This wrapper does NOT automatically sync the GPU copy if the CPU copy is
   ///       updated and vice versa. The one exception to this is nested ManagedArrays
   ///       and managed_ptrs, but only if they are registered via the registerArguments
   ///       method. The factory method make_managed will register arguments passed
   ///       to it automatically. Otherwise, if you wish to keep the CPU and GPU copies
   ///       in sync, you must explicitly modify the object in both the CPU context
   ///       and the GPU context.
   ///    Members of T that are raw pointers need to be initialized correctly with a
   ///       host or device pointer. If it is desired that these be kept in sync,
   ///       pass a ManagedArray to the make_managed function in place of a raw array.
   ///       Or, if this is after the managed_ptr has been constructed, use the same
   ///       ManagedArray in both the CPU and GPU contexts to initialize the raw pointer
   ///       member and then register the ManagedArray with the registerArguments
   ///       method on the managed_ptr. If only a raw array is passed to make_managed,
   ///       accessing that member will be valid in the correct context. To prevent the
   ///       accidental use of them in the wrong context, any methods that access raw
   ///       pointers not initialized in both contexts should be __host__ only or
   ///       __device__ only. Special care should be taken when passing raw pointers
   ///       as arguments to member functions.
   ///    Methods that can be called on the CPU and GPU must be declared with the
   ///       __host__ __device__ specifiers. This includes the constructors being
   ///       used and destructors.
   ///    Raw pointer members still can be used, but they will only be valid on the host.
   ///       To prevent accidentally using them in a device context, any methods that
   ///       access raw pointers should be host only.
   ///
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
         CHAI_HOST_DEVICE constexpr managed_ptr() noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Construct from nullptr.
         /// Initializes the reference count to 0.
         ///
         CHAI_HOST_DEVICE constexpr managed_ptr(std::nullptr_t) noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr& other) noexcept :
#ifdef __CUDACC__
            m_gpu(other.m_gpu),
            m_copyArguments(other.m_copyArguments),
            m_copier(other.m_copier),
            m_deleter(other.m_deleter),
#endif
            m_cpu(other.m_cpu),
            m_numReferences(other.m_numReferences)
         {
#ifndef __CUDA_ARCH__
               incrementReferenceCount();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and increases the reference count.
         ///    U must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         template <typename U>
         CHAI_HOST_DEVICE managed_ptr(managed_ptr<U> const & other) noexcept :
#ifdef __CUDACC__
            m_gpu(other.m_gpu),
            m_copyArguments(other.m_copyArguments),
            m_copier(other.m_copier),
            m_deleter(other.m_deleter),
#endif
            m_cpu(other.m_cpu),
            m_numReferences(other.m_numReferences)
         {
            static_assert(std::is_base_of<T, U>::value ||
                          std::is_convertible<U, T>::value,
                          "Type U must a descendent of or be convertible to type T.");

#ifndef __CUDA_ARCH__
               incrementReferenceCount();
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
            decrementReferenceCount();
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
#ifndef __CUDA_ARCH__
               decrementReferenceCount();
#endif

#ifdef __CUDACC__
               m_gpu = other.m_gpu;
               m_copyArguments = other.m_copyArguments,
               m_copier = other.m_copier,
               m_deleter = other.m_deleter,
#endif
               m_cpu = other.m_cpu;
               m_numReferences = other.m_numReferences;

#ifndef __CUDA_ARCH__
               incrementReferenceCount();
#endif
            }

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Conversion copy assignment operator.
         /// Copies the given managed_ptr and increases the reference count.
         ///    U must be convertible to T.
         ///
         /// @param[in] other The managed_ptr to copy.
         ///
         template<class U>
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr<U>& other) noexcept {
            static_assert(std::is_base_of<T, U>::value ||
                          std::is_convertible<U, T>::value,
                          "Type U must a descendent of or be convertible to type T.");

#ifndef __CUDA_ARCH__
            decrementReferenceCount();
#endif

#ifdef __CUDACC__
            m_gpu = other.m_gpu;
            m_copyArguments = other.m_copyArguments,
            m_copier = other.m_copier,
            m_deleter = other.m_deleter,
#endif
            m_cpu = other.m_cpu;
            m_numReferences = other.m_numReferences;

#ifndef __CUDA_ARCH__
            incrementReferenceCount();
#endif

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

         ///
         /// @author Alan Dayton
         ///
         /// Saves the arguments in order to later call their copy constructor.
         ///
         /// @param[in] args The arguments to save..
         ///
         template <typename... Args>
         CHAI_HOST void registerArguments(Args&&... args) {
            m_copyArguments = (void*) new std::tuple<Args...>(args...);

            m_copier = [] (void* copyArguments) {
               std::tuple<Args...>(*(static_cast<std::tuple<Args...>*>(copyArguments)));
            };

            m_deleter = [] (void* copyArguments) {
               delete static_cast<std::tuple<Args...>*>(copyArguments);
            };
         }

      private:
#ifdef __CUDACC__
         T* m_gpu = nullptr; /// The device pointer
         void* m_copyArguments = nullptr; /// ManagedArrays or managed_ptrs which need the copy constructor called on them
         void (*m_copier)(void*); /// Casts m_copyArguments to the appropriate type and calls the copy constructor
         void (*m_deleter)(void*); /// Casts m_copyArguments to the appropriate type and calls delete
#endif
         T* m_cpu = nullptr; /// The host pointer
         size_t* m_numReferences = nullptr; /// The reference counter

         template <typename U>
         friend class managed_ptr; /// Needed for the converting constructor

         template <typename U, typename... Args>
         friend managed_ptr<U> make_managed(Args&&... args);

         ///
         /// @author Alan Dayton
         ///
         /// Increments the reference count and calls the copy constructor to
         ///    trigger data movement.
         ///
         CHAI_HOST void incrementReferenceCount() {
            if (m_numReferences) {
               (*m_numReferences)++;

               m_copier(m_copyArguments);
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Decrements the reference counter. If the resulting number of references
         ///    is 0, clean up the object.
         ///
         CHAI_HOST void decrementReferenceCount() {
            if (m_numReferences) {
               (*m_numReferences)--;

               if (*m_numReferences == 0) {
                  delete m_numReferences;

                  if (m_deleter) {
                     m_deleter(m_copyArguments);
                  }

                  delete m_cpu;

#ifdef __CUDACC__
                  detail::destroyDevicePtr<<<1, 1>>>(m_gpu);
                  cudaDeviceSynchronize();
#endif
               }
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given host and device pointers.
         ///
         /// @param[in] cpuPtr The host pointer to take ownership of
         /// @param[in] gpuPtr The device pointer to take ownership of
         ///
#ifdef __CUDACC__
         template <typename U, typename V=U>
         CHAI_HOST managed_ptr(U* cpuPtr, V* gpuPtr) :
            m_gpu(gpuPtr),
#else
         template <typename U>
         CHAI_HOST managed_ptr(U* cpuPtr) :
#endif
            m_cpu(cpuPtr),
            m_numReferences(new std::size_t{1})
         {
            static_assert(std::is_base_of<T, U>::value ||
                          std::is_convertible<U, T>::value,
                          "Type U must a descendent of or be convertible to type T.");
#ifdef __CUDACC__
            static_assert(std::is_base_of<T, V>::value ||
                          std::is_convertible<V, T>::value,
                          "Type V must a descendent of or be convertible to type T.");
#endif
         }
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(const managed_ptr<T>& lhs, const managed_ptr<U>& rhs) noexcept {
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(const managed_ptr<T>& lhs, std::nullptr_t) noexcept {
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
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
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
      static_assert(std::is_constructible<T, Args...>::value,
                    "Type T must be constructible with the given arguments.");

      T* cpuPtr = new T(args...);

#ifdef __CUDACC__
      T* gpuPtr = detail::createDevicePtr<T>(args...);
      managed_ptr<T> result(cpuPtr, gpuPtr);
#else
      managed_ptr<T> result(cpuPtr);
#endif

      result.registerArguments(std::forward<Args>(args)...);
      return result;
   }
} // namespace chai

#endif // MANAGED_PTR

