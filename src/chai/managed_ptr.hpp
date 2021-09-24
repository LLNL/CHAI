//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef MANAGED_PTR_H_
#define MANAGED_PTR_H_

#include "chai/config.hpp"

#if defined(CHAI_ENABLE_MANAGED_PTR)

#ifndef CHAI_DISABLE_RM
#include "chai/ArrayManager.hpp"
#endif

#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/Types.hpp"

// Standard libary headers
#include <cstddef>
#include <functional>


namespace chai {
#if defined(CHAI_GPUCC)
   template <typename T>
   CHAI_HOST void destroy_on_device(T* gpuPointer);
#endif

   struct managed_ptr_record {
      managed_ptr_record() = default;

      managed_ptr_record(std::function<bool(Action, ExecutionSpace, void*)> callback) :
         m_callback(callback)
      {
      }

      ExecutionSpace getLastSpace() {
         return m_last_space;
      }

      void set_callback(std::function<bool(Action, ExecutionSpace, void*)> callback) {
         m_callback = callback;
      }

      ExecutionSpace m_last_space = NONE; /// The last space executed in
      std::function<bool(Action, ExecutionSpace, void*)> m_callback; /// Callback to handle events
   };

   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   ///
   /// This wrapper stores both host and device pointers so that polymorphism can be
   ///    used in both contexts with a single API.
   /// The make_managed and make_managed_from_factory functions call new on both the
   ///    host and device so that polymorphism is valid in both contexts. Simply copying
   ///    the bits of an object to the device will not copy the vtable, so new must be
   ///    called on the device.
   ///
   /// Usage Requirements:
   ///    Methods that can be called on the host and/or device must be declared
   ///       with the __host__ and/or __device__ specifiers. This includes constructors
   ///       and destructors. Furthermore, destructors of base and child classes
   ///       must all be declared virtual.
   ///    This wrapper does NOT automatically sync the device object if the host object
   ///       is updated and vice versa. If you wish to keep both instances in sync,
   ///       you must explicitly modify the object in both the host context and the
   ///       device context.
   ///    C-style array members of T need to be initialized correctly with a host or
   ///       device C-style array. If a ManagedArray is passed to the make_managed or
   ///       make_managed_from_factory methods in place of a C-style array, wrap it in
   ///       a call to chai::unpack to extract the C-style arrays contained within the
   ///       ManagedArray. This will pass the extracted host C-style array to the host
   ///       constructor and the extracted device C-style array to the device
   ///       constructor. If it is desired that these host and device C-style arrays be
   ///       kept in sync like the normal behavior of ManagedArray, define a callback
   ///       that maintains a copy of the ManagedArray and upon the ACTION_MOVE event
   ///       calls the copy constructor of that ManagedArray.
   ///    If a C-style array is passed to make_managed, accessing that member will be
   ///       valid only in the correct context. To prevent the accidental use of that
   ///       member in the wrong context, any methods that access it should be __host__
   ///       only or __device__ only. Special care should be taken when passing C-style
   ///       arrays as arguments to member functions.
   ///    The same restrictions for C-style array members also apply to raw pointer
   ///       members. If a managed_ptr is passed to the make_managed or
   ///       make_managed_from_factory methods in place of a raw pointer, wrap it in
   ///       a call to chai::unpack to extract the raw pointers contained within the
   ///       managed_ptr. This will pass the extracted host pointer to the host
   ///       constructor and the extracted device pointer to the device constructor.
   ///       If it is desired that these host and device pointers be kept in sync,
   ///       define a callback that maintains a copy of the managed_ptr and upon the
   ///       ACTION_MOVE event call the copy constructor of that managed_ptr.
   ///    Again, if a raw pointer is passed to make_managed, accessing that member will
   ///       only be valid in the correct context. Take care when passing raw pointers
   ///       as arguments to member functions.
   ///    Be aware that CHAI checks every CUDA API call for GPU errors by default. To
   ///       turn off GPU error checking, pass -DCHAI_ENABLE_GPU_ERROR_CHECKING=OFF as
   ///       an argument to cmake when building CHAI. To turn on synchronization after
   ///       every kernel, call ArrayManager::getInstance()->enableDeviceSynchronize().
   ///       Alternatively, call cudaDeviceSynchronize() after any call to make_managed,
   ///       make_managed_from_factory, or managed_ptr::free, and check the return code
   ///       for errors. If your code crashes in the constructor/destructor of T, then
   ///       it is recommended to turn on this synchronization. For example, the
   ///       constructor of T might run out of per-thread stack space on the GPU. If
   ///       that happens, you can increase the device limit of per-thread stack space.
   ///
   template <typename T>
   class managed_ptr {
      public:
         using element_type = T;

         ///
         /// @author Alan Dayton
         ///
         /// Default constructor.
         ///
         constexpr managed_ptr() noexcept = default;

         ///
         /// @author Alan Dayton
         ///
         /// Construct from nullptr.
         ///
         CHAI_HOST_DEVICE constexpr managed_ptr(std::nullptr_t) noexcept {}

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given pointers. U* must be convertible
         ///    to T*.
         ///
         /// @pre spaces.size() == pointers.size()
         ///
         /// @param[in] spaces A list of execution spaces
         /// @param[in] pointers A list of pointers to take ownership of
         ///
         template <typename U>
         managed_ptr(std::initializer_list<ExecutionSpace> spaces,
                     std::initializer_list<U*> pointers) :
            m_cpu_pointer(nullptr),
            m_gpu_pointer(nullptr),
            m_pointer_record(new managed_ptr_record())
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

            // TODO: In c++14 convert to a static_assert
            if (spaces.size() != pointers.size()) {
               printf("[CHAI] WARNING: The number of spaces is different than the number of pointers given!\n");
            }

            int i = 0;

            for (const auto& space : spaces) {
               switch (space) {
                  case CPU:
                     m_cpu_pointer = pointers.begin()[i++];
                     break;
#if defined(CHAI_GPUCC)
                  case GPU:
                     m_gpu_pointer = pointers.begin()[i++];
                     break;
#endif
                  default:
                     ++i;
                     printf("[CHAI] WARNING: Execution space not supported by chai::managed_ptr!\n");
                     break;
               }
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Constructs a managed_ptr from the given pointers and callback function.
         ///    U* must be convertible to T*.
         ///
         /// @pre spaces.size() == pointers.size()
         ///
         /// @param[in] spaces A list of execution spaces
         /// @param[in] pointers A list of pointers to take ownership of
         /// @param[in] callback The user defined callback to call on trigger events
         ///
         template <typename U>
         CHAI_HOST managed_ptr(std::initializer_list<ExecutionSpace> spaces,
                               std::initializer_list<U*> pointers,
                               std::function<bool(Action, ExecutionSpace, void*)> callback) :
            m_cpu_pointer(nullptr),
            m_gpu_pointer(nullptr),
            m_pointer_record(new managed_ptr_record(callback))
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

            // TODO: In c++14 convert to a static_assert
            if (spaces.size() != pointers.size()) {
               printf("[CHAI] WARNING: The number of spaces is different than the number of pointers given.\n");
            }

            int i = 0;

            for (const auto& space : spaces) {
               switch (space) {
                  case CPU:
                     m_cpu_pointer = pointers.begin()[i++];
                     break;
#if defined(CHAI_GPUCC)
                  case GPU:
                     m_gpu_pointer = pointers.begin()[i++];
                     break;
#endif
                  default:
                     ++i;
                     printf("[CHAI] WARNING: Execution space not supported by chai::managed_ptr!\n");
                     break;
               }
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         /// Constructs a copy of the given managed_ptr and if the execution space is
         ///    different from the last space the given managed_ptr was used in, calls
         ///    the user defined callback with ACTION_MOVE for each of the execution
         ///    spaces.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr& other) noexcept :
            m_cpu_pointer(other.m_cpu_pointer),
            m_gpu_pointer(other.m_gpu_pointer),
            m_pointer_record(other.m_pointer_record)
         {
#if !defined(CHAI_DEVICE_COMPILE)
            move();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Converting constructor.
         /// Constructs a copy of the given managed_ptr and if the execution space is
         ///    different from the last space the given managed_ptr was used in, calls
         ///    the user defined callback with ACTION_MOVE for each of the execution
         ///    spaces. U* must be convertible to T*.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template <typename U>
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr<U>& other) noexcept :
            m_cpu_pointer(other.m_cpu_pointer),
            m_gpu_pointer(other.m_gpu_pointer),
            m_pointer_record(other.m_pointer_record)
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

#if !defined(CHAI_DEVICE_COMPILE)
            move();
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Aliasing constructor.
         /// Has the same ownership information as other, but holds different pointers.
         ///
         /// @pre spaces.size() == pointers.size()
         ///
         /// @param[in] other The managed_ptr to copy ownership information from
         /// @param[in] spaces A list of execution spaces
         /// @param[in] pointers A list of pointers to maintain a reference to
         ///
         template <typename U>
         CHAI_HOST managed_ptr(const managed_ptr<U>& other,
                               std::initializer_list<ExecutionSpace> spaces,
                               std::initializer_list<T*> pointers) noexcept :
            m_pointer_record(other.m_pointer_record)
         {
            // TODO: In c++14 convert to a static_assert
            if (spaces.size() != pointers.size()) {
               printf("[CHAI] WARNING: The number of spaces is different than the number of pointers given.\n");
            }

            int i = 0;

            for (const auto& space : spaces) {
               switch (space) {
                  case CPU:
                     m_cpu_pointer = pointers.begin()[i++];
                     break;
#if defined(CHAI_GPUCC)
                  case GPU:
                     m_gpu_pointer = pointers.begin()[i++];
                     break;
#endif
                  default:
                     ++i;
                     printf("[CHAI] WARNING: Execution space not supported by chai::managed_ptr!\n");
                     break;
               }
            }

            move();
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy assignment operator.
         /// Copies the given managed_ptr and if the execution space is different from
         ///    the last space the given managed_ptr was used in, calls the user defined
         ///    callback with ACTION_MOVE for each of the execution spaces.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr& other) noexcept {
            if (this != &other) {
               m_cpu_pointer = other.m_cpu_pointer;
               m_gpu_pointer = other.m_gpu_pointer;
               m_pointer_record = other.m_pointer_record;

#if !defined(CHAI_DEVICE_COMPILE)
               move();
#endif
            }

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Conversion copy assignment operator.
         /// Copies the given managed_ptr and if the execution space is different from
         ///    the last space the given managed_ptr was used in, calls the user defined
         ///    callback with ACTION_MOVE for each of the execution spaces. U* must be
         ///    convertible to T*.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template <typename U>
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr<U>& other) noexcept {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

            m_cpu_pointer = other.m_cpu_pointer;
            m_gpu_pointer = other.m_gpu_pointer;
            m_pointer_record = other.m_pointer_record;

#if !defined(CHAI_DEVICE_COMPILE)
            move();
#endif

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* get() const {
#if !defined(CHAI_DEVICE_COMPILE)
            move();
            return m_cpu_pointer;
#else
            return m_gpu_pointer;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the pointer corresponding to the given execution space.
         ///
         /// @param[in] space The execution space
         /// @param[in] move Whether or not to trigger the move event (default is true)
         ///
         CHAI_HOST inline T* get(const ExecutionSpace space, const bool move=true) const {
            if (move) {
               this->move();
            }

            switch (space) {
               case CPU:
                  return m_cpu_pointer;
#if defined(CHAI_GPUCC)
               case GPU:
                  return m_gpu_pointer;
#endif
               default:
                  return nullptr;
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T* operator->() const {
#if !defined(CHAI_DEVICE_COMPILE)
            return m_cpu_pointer;
#else
            return m_gpu_pointer;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU reference depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T& operator*() const {
#if !defined(CHAI_DEVICE_COMPILE)
            return *m_cpu_pointer;
#else
            return *m_gpu_pointer;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns true if the contained pointer is not nullptr, false otherwise.
         ///
         CHAI_HOST_DEVICE inline explicit operator bool() const noexcept {
            return get() != nullptr;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Sets the callback, which can be used to handle specific actions.
         /// The copy constructors and copy assignment operators call the callback with
         ///    ACTION_MOVE if the execution space has changed since the managed_ptr was
         ///    last used. A common use case for this is to call the copy constructor
         ///    of class members that are ManagedArrays to trigger data movement. The
         ///    free method calls the user provided callback with ACTION_FREE in each of
         ///    the execution spaces with the pointers from each space. This can be used
         ///    to provide a custom deleter operation. If freeing anything other than the
         ///    actual object pointers, do that when the ExecutionSpace is NONE. The
         ///    callback should return true if the event has been handled (i.e. if a
         ///    callback is provided that only cleans up the device pointer, it should
         ///    return true in that case and false in every other case).
         ///
         /// @param[in] callback The callback to call when certain actions occur
         ///
         CHAI_HOST void set_callback(std::function<bool(Action, ExecutionSpace, void*)> callback) {
            if (m_pointer_record) {
               m_pointer_record->set_callback(callback);
            }
            else {
               printf("[CHAI] WARNING: No callback is allowed for managed_ptr that does not contain a valid pointer (i.e. the default or nullptr constructor was used)!\n");
            }
         }

         ///
         /// @author Alan Dayton
         ///
         /// If a user defined callback has been provided, calls it with the ACTION_FREE
         ///    event in each execution space. If the callback does not handle an event
         ///    or a callback is not provided, this method calls delete on the host
         ///    and device pointers.
         ///
         CHAI_HOST void free() {
            if (m_pointer_record) {
               if (m_pointer_record->m_callback) {
                  // Destroy device pointer first to take advantage of asynchrony
                  for (int space = NUM_EXECUTION_SPACES-1; space >= NONE; --space) {
                     ExecutionSpace execSpace = static_cast<ExecutionSpace>(space);
                     T* pointer = get(execSpace, false);

                     using T_non_const = typename std::remove_const<T>::type;

                     // We can use const_cast because can managed_ptr can only
                     // be constructed with non const pointers.
                     T_non_const* temp = const_cast<T_non_const*>(pointer);
                     void* voidPointer = static_cast<void*>(temp);

                     if (!m_pointer_record->m_callback(ACTION_FREE,
                                                       execSpace,
                                                       voidPointer)) {
                        switch (execSpace) {
                           case CPU:
                              delete pointer;
                              m_cpu_pointer = nullptr;
                              break;
#if defined(CHAI_GPUCC)
                           case GPU:
                           {
                              if (pointer) {
                                 destroy_on_device(temp);
                                 m_gpu_pointer = nullptr;
#ifndef CHAI_DISABLE_RM
                                 if (ArrayManager::getInstance()->deviceSynchronize()) {
                                    synchronize();
                                 }
#endif
                              }

                              break;
                           }
#endif
                           default:
                              break;
                        }
                     }
                  }
               }
               else {
                  // Destroy device pointer first to take advantage of asynchrony
                  for (int space = NUM_EXECUTION_SPACES-1; space >= NONE; --space) {
                     ExecutionSpace execSpace = static_cast<ExecutionSpace>(space);
                     T* pointer = get(execSpace, false);

                     switch (execSpace) {
                        case CPU:
                           delete pointer;
                           m_cpu_pointer = nullptr;
                           break;
#if defined(CHAI_GPUCC)
                        case GPU:
                        {
                           if (pointer) {
                              destroy_on_device(pointer);
                              m_gpu_pointer = nullptr;
#ifndef CHAI_DISABLE_RM
                              if (ArrayManager::getInstance()->deviceSynchronize()) {
                                 synchronize();
                              }
#endif
                           }

                           break;
                        }
#endif
                        default:
                           break;
                     }
                  }
               }

               delete m_pointer_record;
               m_pointer_record = nullptr;
            }
         }

      private:
         T* m_cpu_pointer = nullptr; /// The CPU pointer
         T* m_gpu_pointer = nullptr; /// The GPU pointer
         managed_ptr_record* m_pointer_record = nullptr; /// The pointer record

         /// Needed for the converting constructor
         template <typename U>
         friend class managed_ptr;

         /// Needed to use the make_managed API
         template <typename U,
                   typename... Args>
         friend CHAI_HOST managed_ptr<U> make_managed(Args... args);

         ///
         /// @author Alan Dayton
         ///
         /// If the execution space has changed, calls the user provided callback
         ///    with the ACTION_MOVE event.
         ///
         CHAI_HOST void move() const {
#ifndef CHAI_DISABLE_RM
            if (m_pointer_record) {
               ExecutionSpace newSpace = ArrayManager::getInstance()->getExecutionSpace();

               if (newSpace != NONE && newSpace != m_pointer_record->getLastSpace()) {
                  m_pointer_record->m_last_space = newSpace;

                  if (m_pointer_record->m_callback) {
                     for (int space = NONE; space < NUM_EXECUTION_SPACES; ++space) {
                        ExecutionSpace execSpace = static_cast<ExecutionSpace>(space);

                        T* pointer = get(execSpace, false);

                        using T_non_const = typename std::remove_const<T>::type;

                        // We can use const_cast because can managed_ptr can only
                        // be constructed with non const pointers.
                        T_non_const* temp = const_cast<T_non_const*>(pointer);

                        void* voidPointer = static_cast<void*>(temp);

                        m_pointer_record->m_callback(ACTION_MOVE, execSpace, voidPointer);
                     }
                  }
               }
            }
#endif
         }
   };

   ///
   /// @author Alan Dayton
   ///
   /// A wrapper used by the make_managed family of functions to indicate when
   /// the internal pointers contained by a ManagedArray should be extracted.
   /// It is not intended to be used directly, but rather created by unpack.
   ///
   template <typename T>
   class ManagedArrayUnpacker {
      public:
         CHAI_HOST ManagedArrayUnpacker() = delete;

         ///
         /// @author Alan Dayton
         ///
         /// Constructor
         ///
         /// @param[in] arg The ManagedArray to unpack
         ///
         /// @return a new instance of ManagedArrayUnpacker
         ///
         explicit CHAI_HOST ManagedArrayUnpacker(const ManagedArray<T>& arg)
            : m_array{arg}
         {}

         ///
         /// @author Alan Dayton
         ///
         /// Unpacks the data
         ///
         /// @return the unpacked data
         ///
         CHAI_HOST_DEVICE T* data() const { return m_array.data(); }

      private:
         ManagedArray<T> m_array = nullptr; //!< The ManagedArray to unpack
   };

   ///
   /// @author Alan Dayton
   ///
   /// A wrapper used by the make_managed family of functions to indicate when
   /// the internal pointers contained by a managed_ptr should be extracted.
   /// It is not intended to be used directly, but rather created by unpack.
   ///
   template <typename T>
   class managed_ptr_unpacker {
      public:
         CHAI_HOST managed_ptr_unpacker() = delete;

         ///
         /// @author Alan Dayton
         ///
         /// Constructor
         ///
         /// @param[in] arg The managed_ptr to unpack
         ///
         /// @return a new instance of managed_ptr_unpacker
         ///
         explicit CHAI_HOST managed_ptr_unpacker(const managed_ptr<T>& arg)
            : m_managed_ptr{arg}
         {}

         ///
         /// @author Alan Dayton
         ///
         /// Unpacks the data
         ///
         /// @return the unpacked data
         ///
         CHAI_HOST_DEVICE T* get() const { return m_managed_ptr.get(); }

      private:
         managed_ptr<T> m_managed_ptr = nullptr; //!< The managed_ptr to unpack
   };

   namespace detail {

      ///
      /// @author Alan Dayton
      ///
      /// This implementation of getRawPointers handles every non-CHAI type.
      ///
      /// @param[in] arg The non-CHAI type, which will simply be returned
      ///
      /// @return arg
      ///
      template <typename T>
      CHAI_HOST_DEVICE T processArguments(const T& arg) {
         return arg;
      }

      ///
      /// @author Alan Dayton
      ///
      /// This implementation of getRawPointers handles the CHAI ManagedArray type.
      ///
      /// @param[in] arg The ManagedArray from which to extract a raw pointer
      ///
      /// @return arg cast to a raw pointer
      ///
      template <typename T>
      CHAI_HOST_DEVICE T* processArguments(const ManagedArrayUnpacker<T>& arg) {
         return arg.data();
      }

      ///
      /// @author Alan Dayton
      ///
      /// This implementation of getRawPointers handles the CHAI managed_ptr type.
      /// The managed_ptr type is not implicitly convertible to a raw pointer, so
      ///    when using the make_managed API, it is necessary to pull the raw pointers
      ///    out of the managed_ptr.
      ///
      /// @param[in] arg The managed_ptr from which to extract a raw pointer
      ///
      /// @return a raw pointer acquired from arg
      ///
      template <typename T>
      CHAI_HOST_DEVICE T* processArguments(const managed_ptr_unpacker<T>& arg) {
         return arg.get();
      }

#if defined(CHAI_GPUCC)

      ///
      /// @author Alan Dayton
      ///
      /// Creates a new T on the device.
      ///
      /// @param[out] gpuPointer Used to return the device pointer to the new T
      /// @param[in]  args The arguments to T's constructor
      ///
      /// @note Cannot capture argument packs in an extended device lambda,
      ///       so explicit kernel is needed.
      ///
      template <typename T,
                typename... Args>
      __global__ void make_on_device(T** gpuPointer, Args... args)
      {
         *gpuPointer = new T(processArguments(args)...);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Creates a new object on the device by calling the given factory method.
      ///
      /// @param[out] gpuPointer Used to return the device pointer to the new object
      /// @param[in]  f The factory method (must be a __device__ or __host__ __device__
      ///                method
      /// @param[in]  args The arguments to the factory method
      ///
      /// @note Cannot capture argument packs in an extended device lambda,
      ///       so explicit kernel is needed.
      ///
      template <typename T,
                typename F,
                typename... Args>
      __global__ void make_on_device_from_factory(T** gpuPointer, F f, Args... args)
      {
         *gpuPointer = f(processArguments(args)...);
      }

      ///
      /// @author Alan Dayton
      ///
      /// Destroys the device pointer.
      ///
      /// @param[out] gpuPointer The device pointer to call delete on
      ///
      template <typename T>
      __global__ void destroy_on_device(T* gpuPointer)
      {
         delete gpuPointer;
      }

#endif

      // Adapted from "The C++ Programming Language," Fourth Edition,
      // by Bjarne Stroustrup, pp. 814-816
      // Used to determine if a functor is callable with the given arguments
      struct substitution_failure {};

      template <typename T>
      struct substitution_succeeded : std::true_type {};

      template<>
      struct substitution_succeeded<substitution_failure> : std::false_type {};

      template <typename F, typename... Args>
      struct is_invocable_impl {
         private:
            template <typename X, typename... Ts>
            static auto check(X const& x, Ts&&... ts) -> decltype(x(ts...));
            static substitution_failure check(...);
         public:
            using type = decltype(check(std::declval<F>(), std::declval<Args>()...));
      };

      template <typename F, typename... Args>
      struct is_invocable : substitution_succeeded<typename is_invocable_impl<F, Args...>::type> {};
   } // namespace detail

   ///
   /// @author Alan Dayton
   ///
   /// Unpacks the pointers contained in the ManagedArray and passes them to the
   /// corresponding spaces.
   ///
   /// @param[in] arg The ManagedArray to unpack
   ///
   /// @return A wrapper used by make_managed for unpacking the internal pointers
   ///         in the correct space
   ///
   template <typename T>
   CHAI_HOST ManagedArrayUnpacker<T> unpack(const ManagedArray<T>& arg) {
      return ManagedArrayUnpacker<T>(arg);
   }

   ///
   /// @author Alan Dayton
   ///
   /// Unpacks the pointers contained in the managed_ptr and passes them to the
   /// corresponding spaces.
   ///
   /// @param[in] arg The managed_ptr to unpack
   ///
   /// @return A wrapper used by make_managed for unpacking the internal pointers
   ///         in the correct space
   ///
   template <typename T>
   CHAI_HOST managed_ptr_unpacker<T> unpack(const managed_ptr<T>& arg) {
      return managed_ptr_unpacker<T>(arg);
   }

   ///
   /// @author Alan Dayton
   ///
   /// Creates a new T on the host.
   /// Sets the execution space to the CPU so that ManagedArrays and managed_ptrs
   ///    are moved to the host as necessary.
   ///
   /// @param[in]  args The arguments to T's constructor
   ///
   /// @return The host pointer to the new T
   ///
   template <typename T,
             typename... Args>
   CHAI_HOST T* make_on_host(Args&&... args) {
#ifndef CHAI_DISABLE_RM
      // Get the ArrayManager and save the current execution space
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();

      // Set the execution space so that ManagedArrays and managed_ptrs
      // are handled properly
      arrayManager->setExecutionSpace(CPU);
#endif

      // Create on the host
      T* cpuPointer = new T(detail::processArguments(args)...);

#ifndef CHAI_DISABLE_RM
      // Set the execution space back to the previous value
      arrayManager->setExecutionSpace(currentSpace);
#endif

      // Return the CPU pointer
      return cpuPointer;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Calls a factory method to create a new object on the host.
   /// Sets the execution space to the CPU so that ManagedArrays and managed_ptrs
   ///    are moved to the host as necessary.
   ///
   /// @param[in]  f    The factory method
   /// @param[in]  args The arguments to the factory method
   ///
   /// @return The host pointer to the new object
   ///
   template <typename T,
             typename F,
             typename... Args>
   CHAI_HOST T* make_on_host_from_factory(F f, Args&&... args) {
#ifndef CHAI_DISABLE_RM
      // Get the ArrayManager and save the current execution space
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();

      // Set the execution space so that ManagedArrays and managed_ptrs
      // are handled properly
      arrayManager->setExecutionSpace(CPU);
#endif

      // Create the object on the device
      T* cpuPointer = f(args...);

#ifndef CHAI_DISABLE_RM
      // Set the execution space back to the previous value
      arrayManager->setExecutionSpace(currentSpace);
#endif

      // Return the GPU pointer
      return cpuPointer;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Destroys the host pointer.
   ///
   /// @param[out] cpuPointer The host pointer to clean up
   ///
   template <typename T>
   CHAI_HOST void destroy_on_host(T* cpuPointer) {
      delete cpuPointer;
   }

#if defined(CHAI_GPUCC)

   ///
   /// @author Alan Dayton
   ///
   /// Creates a new T on the device.
   ///
   /// @param[in]  args The arguments to T's constructor
   ///
   /// @return The device pointer to the new T
   ///
   template <typename T,
             typename... Args>
   CHAI_HOST T* make_on_device(Args... args) {
#ifndef CHAI_DISABLE_RM
      // Get the ArrayManager and save the current execution space
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();

      // Set the execution space so that ManagedArrays and managed_ptrs
      // are handled properly
      arrayManager->setExecutionSpace(GPU);
#endif

      // Allocate space on the GPU to hold the pointer to the new object
      T** gpuBuffer;
      gpuMalloc((void**)(&gpuBuffer), sizeof(T*));

      // Create the object on the device
#if defined(__CUDACC__)
      detail::make_on_device<<<1, 1>>>(gpuBuffer, args...);
#elif defined(__HIPCC__)
      hipLaunchKernelGGL(detail::make_on_device, 1, 1, 0, 0, gpuBuffer, args...);
#endif

#ifndef CHAI_DISABLE_RM
      if (ArrayManager::getInstance()->deviceSynchronize()) {
         synchronize();
      }
#endif

      // Allocate space on the CPU for the pointer and copy the pointer to the CPU
      T** cpuBuffer = (T**) malloc(sizeof(T*));
      gpuMemcpy(cpuBuffer, gpuBuffer, sizeof(T*), gpuMemcpyDeviceToHost);

      // Get the GPU pointer
      T* gpuPointer = cpuBuffer[0];

      // Free the host and device buffers
      free(cpuBuffer);
      gpuFree(gpuBuffer);

#ifndef CHAI_DISABLE_RM
      // Set the execution space back to the previous value
      arrayManager->setExecutionSpace(currentSpace);
#endif

      // Return the GPU pointer
      return gpuPointer;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Calls a factory method to create a new object on the device.
   ///
   /// @param[in]  f    The factory method
   /// @param[in]  args The arguments to the factory method
   ///
   /// @return The device pointer to the new object
   ///
   template <typename T,
             typename F,
             typename... Args>
   CHAI_HOST T* make_on_device_from_factory(F f, Args&&... args) {
#ifndef CHAI_DISABLE_RM
      // Get the ArrayManager and save the current execution space
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();

      // Set the execution space so that chai::ManagedArrays and
      // chai::managed_ptrs are handled properly
      arrayManager->setExecutionSpace(GPU);
#endif

      // Allocate space on the GPU to hold the pointer to the new object
      T** gpuBuffer;
      gpuMalloc((void**)(&gpuBuffer), sizeof(T*));

      // Create the object on the device
#if defined(__CUDACC__)
      detail::make_on_device_from_factory<T><<<1, 1>>>(gpuBuffer, f, args...);
#elif defined(__HIPCC__)
      hipLaunchKernelGGL(detail::make_on_device_from_factory, 1, 1, 0, 0, gpuBuffer, f, args...);
#endif

#ifndef CHAI_DISABLE_RM
      if (ArrayManager::getInstance()->deviceSynchronize()) {
         synchronize();
      }
#endif

      // Allocate space on the CPU for the pointer and copy the pointer to the CPU
      T** cpuBuffer = (T**) malloc(sizeof(T*));
      gpuMemcpy(cpuBuffer, gpuBuffer, sizeof(T*), gpuMemcpyDeviceToHost);

      // Get the GPU pointer
      T* gpuPointer = cpuBuffer[0];

      // Free the host and device buffers
      free(cpuBuffer);
      gpuFree(gpuBuffer);

#ifndef CHAI_DISABLE_RM
      // Set the execution space back to the previous value
      arrayManager->setExecutionSpace(currentSpace);
#endif

      // Return the GPU pointer
      return gpuPointer;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Destroys the device pointer.
   ///
   /// @param[out] gpuPointer The device pointer to clean up
   ///
   template <typename T>
   CHAI_HOST void destroy_on_device(T* gpuPointer) {
#if defined(__CUDACC__)
      detail::destroy_on_device<<<1, 1>>>(gpuPointer);
#elif defined(__HIPCC__)
      hipLaunchKernelGGL(detail::destroy_on_device, 1, 1, 0, 0, gpuPointer);
#endif
   }

#endif

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] args The arguments to T's constructor
   ///
   template <typename T,
             typename... Args>
   CHAI_HOST managed_ptr<T> make_managed(Args... args) {
#if defined(CHAI_GPUCC)
      // Construct on the GPU first to take advantage of asynchrony
      T* gpuPointer = make_on_device<T>(args...);
#endif

      // Construct on the CPU
      T* cpuPointer = make_on_host<T>(args...);

      // Construct and return the managed_ptr
#if defined(CHAI_GPUCC)
      return managed_ptr<T>({CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>({CPU}, {cpuPointer});
#endif
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] f The factory function that will create the object
   /// @param[in] args The arguments to the factory function
   ///
   template <typename T,
             typename F,
             typename... Args>
   CHAI_HOST managed_ptr<T> make_managed_from_factory(F&& f, Args&&... args) {
      static_assert(detail::is_invocable<F, Args...>::value,
                    "F is not invocable with the given arguments.");

      static_assert(std::is_pointer<typename std::result_of<F(Args...)>::type>::value,
                    "F does not return a pointer.");

      using R = typename std::remove_pointer<typename std::result_of<F(Args...)>::type>::type;

      static_assert(std::is_convertible<R*, T*>::value,
                    "F does not return a pointer that is convertible to T*.");

#if defined(CHAI_GPUCC)
      // Construct on the GPU first to take advantage of asynchrony
      T* gpuPointer = make_on_device_from_factory<R>(f, args...);
#endif

      // Construct on the CPU
      T* cpuPointer = make_on_host_from_factory<R>(f, args...);

      // Construct and return the managed_ptr
#if defined(CHAI_GPUCC)
      return managed_ptr<T>({CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>({CPU}, {cpuPointer});
#endif
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using static_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using static_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> static_pointer_cast(const managed_ptr<U>& other) noexcept {
      T* cpuPointer = static_cast<T*>(other.get());

#if defined(CHAI_GPUCC)
      T* gpuPointer = static_cast<T*>(other.get(GPU, false));

      return managed_ptr<T>(other, {CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>(other, {CPU}, {cpuPointer});
#endif
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using dynamic_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using dynamic_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> dynamic_pointer_cast(const managed_ptr<U>& other) noexcept {
      T* cpuPointer = dynamic_cast<T*>(other.get());

#if defined(CHAI_GPUCC)
      T* gpuPointer = nullptr;

      if (cpuPointer) {
         gpuPointer = static_cast<T*>(other.get(GPU, false));
      }

      return managed_ptr<T>(other, {CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>(other, {CPU}, {cpuPointer});
#endif
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using const_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using const_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> const_pointer_cast(const managed_ptr<U>& other) noexcept {
      T* cpuPointer = const_cast<T*>(other.get());

#if defined(CHAI_GPUCC)
      T* gpuPointer = const_cast<T*>(other.get(GPU, false));

      return managed_ptr<T>(other, {CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>(other, {CPU}, {cpuPointer});
#endif
   }

   ///
   /// @author Alan Dayton
   ///
   /// Makes a new managed_ptr that shares ownership with the given managed_ptr, but
   ///    the underlying pointer is converted using reinterpret_cast.
   ///
   /// @param[in] other The managed_ptr to share ownership with and whose pointer to
   ///                      convert using reinterpret_cast
   ///
   template <typename T, typename U>
   CHAI_HOST managed_ptr<T> reinterpret_pointer_cast(const managed_ptr<U>& other) noexcept {
      T* cpuPointer = reinterpret_cast<T*>(other.get());

#if defined(CHAI_GPUCC)
      T* gpuPointer = reinterpret_cast<T*>(other.get(GPU, false));

      return managed_ptr<T>(other, {CPU, GPU}, {cpuPointer, gpuPointer});
#else
      return managed_ptr<T>(other, {CPU}, {cpuPointer});
#endif
   }

   /// Comparison operators

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to compare
   /// @param[in] rhs The second managed_ptr to compare
   ///
   template <typename T, typename U>
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
   template <typename T, typename U>
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
   template <typename T>
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
   template <typename T>
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
   template <typename T>
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
   template <typename T>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator!=(std::nullptr_t, const managed_ptr<T>& rhs) noexcept {
      return nullptr != rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to swap
   /// @param[in] rhs The second managed_ptr to swap
   ///
   template <typename T>
   void swap(managed_ptr<T>& lhs, managed_ptr<T>& rhs) noexcept {
      std::swap(lhs.m_cpu_pointer, rhs.m_cpu_pointer);
      std::swap(lhs.m_gpu_pointer, rhs.m_gpu_pointer);
      std::swap(lhs.m_pointer_record, rhs.m_pointer_record);
   }
} // namespace chai

#else // defined(CHAI_ENABLE_MANAGED_PTR)

#error CHAI must be configured with -DCHAI_ENABLE_MANAGED_PTR=ON to use managed_ptr! \
       If CHAI_ENABLE_MANAGED_PTR is defined as a macro, it is safe to include managed_ptr.hpp.

#endif // defined(CHAI_ENABLE_MANAGED_PTR)

#endif // MANAGED_PTR

