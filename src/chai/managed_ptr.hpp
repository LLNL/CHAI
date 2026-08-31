//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_MANAGED_PTR_HPP
#define CHAI_MANAGED_PTR_HPP

#include "chai/config.hpp"

#if defined(CHAI_ENABLE_MANAGED_PTR)

#if !defined(CHAI_DISABLE_RM) || defined(CHAI_THIN_GPU_ALLOCATE)
#include "chai/ArrayManager.hpp"
#endif

#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/Types.hpp"

#include "umpire/ResourceManager.hpp"

// Standard libary headers
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>


namespace chai {
   template <typename T>
   CHAI_HOST void destroy_on_host(T* cpuPointer);

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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
      std::function<void(void*)> m_cpu_destroy; /// Concrete destroyer for CPU allocations
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      std::function<void(void*)> m_gpu_destroy; /// Concrete destroyer for GPU allocations
#endif
   };

   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   ///
   /// This wrapper stores both host and device pointers so that polymorphism can be
   ///    used in both contexts with a single API.
   /// The make_managed function calls new on both the host and device so that
   ///    polymorphism is valid in both contexts. Simply copying the bits of an
   ///    object to the device will not copy the vtable, so new must be called
   ///    on the device.
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
   ///       device C-style array. If a ManagedArray is passed to the make_managed
   ///       function in place of a C-style array, wrap it in a call to chai::unpack to
   ///       extract the C-style arrays contained within the ManagedArray. This will
   ///       pass the extracted host C-style array to the host constructor and the
   ///       extracted device C-style array to the device constructor. If it is desired
   ///       that these host and device C-style arrays be kept in sync like the normal
   ///       behavior of ManagedArray, define a callback that maintains a copy of the
   ///       ManagedArray and upon the ACTION_MOVE event calls the copy constructor of
   ///       that ManagedArray.
   ///    If a C-style array is passed to make_managed, accessing that member will be
   ///       valid only in the correct context. To prevent the accidental use of that
   ///       member in the wrong context, any methods that access it should be __host__
   ///       only or __device__ only. Special care should be taken when passing C-style
   ///       arrays as arguments to member functions.
   ///    The same restrictions for C-style array members also apply to raw pointer
   ///       members. If a managed_ptr is passed to the make_managed function in place of
   ///       a raw pointer, wrap it in a call to chai::unpack to extract the raw pointers
   ///       contained within the managed_ptr. This will pass the extracted host pointer
   ///       to the host constructor and the extracted device pointer to the device
   ///       constructor. If it is desired that these host and device pointers be kept in
   ///       sync, define a callback that maintains a copy of the managed_ptr and upon the
   ///       ACTION_MOVE event call the copy constructor of that managed_ptr.
   ///    Again, if a raw pointer is passed to make_managed, accessing that member will
   ///       only be valid in the correct context. Take care when passing raw pointers
   ///       as arguments to member functions.
   ///    Be aware that CHAI checks every CUDA API call for GPU errors by default. To
   ///       turn off GPU error checking, pass -DCHAI_ENABLE_GPU_ERROR_CHECKING=OFF as
   ///       an argument to cmake when building CHAI. To turn on synchronization after
   ///       every kernel, set the appropriate environment variable (e.g. CUDA_LAUNCH_BLOCKING or HIP_LAUNCH_BLOCKING).
   ///       Alternatively, call cudaDeviceSynchronize() after any call to make_managed
   ///       or managed_ptr::free, and check the return code for errors. If your code
   ///       crashes in the constructor/destructor of T, then it is recommended to turn
   ///       on this synchronization for debugging. For example, the constructor of T
   ///       might run out of per-thread stack space on the GPU. If that happens, you
   ///       can increase the device limit of per-thread stack space.
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_pointer(nullptr),
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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

            m_pointer_record->m_cpu_destroy = [] (void* pointer) {
               destroy_on_host(static_cast<U*>(pointer));
            };
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_pointer_record->m_gpu_destroy = [] (void* pointer) {
               destroy_on_device(static_cast<U*>(pointer));
            };
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_pointer(nullptr),
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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

            m_pointer_record->m_cpu_destroy = [] (void* pointer) {
               destroy_on_host(static_cast<U*>(pointer));
            };
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_pointer_record->m_gpu_destroy = [] (void* pointer) {
               destroy_on_device(static_cast<U*>(pointer));
            };
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_pointer(other.m_gpu_pointer),
#endif
            m_pointer_record(other.m_pointer_record)
         {
#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
                return;
            }
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_pointer(other.m_gpu_pointer),
#endif
            m_pointer_record(other.m_pointer_record)
         {
            static_assert(std::is_convertible<U*, T*>::value,
                          "U* must be convertible to T*.");

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
                return;
            }
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               m_gpu_pointer = other.m_gpu_pointer;
#endif
               m_pointer_record = other.m_pointer_record;

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
                  return *this;
               }
#endif
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_pointer = other.m_gpu_pointer;
#endif
            m_pointer_record = other.m_pointer_record;

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
               return *this;
            }
#endif

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
#if defined(CHAI_DEVICE_COMPILE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            return m_gpu_pointer;
#else

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
               return m_gpu_pointer;
            }
#endif
            move();
#endif

            return m_cpu_pointer;
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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
#if defined(CHAI_DEVICE_COMPILE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            return m_gpu_pointer;
#else

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
               return m_gpu_pointer;
            }
#endif
            move();
#endif

            return m_cpu_pointer;
#endif
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU reference depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline T& operator*() const {
#if defined(CHAI_DEVICE_COMPILE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            return *m_gpu_pointer;
#else

#if !defined(CHAI_DEVICE_COMPILE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (chai::ArrayManager::getInstance()->isGPUSimMode()) {
               return *m_gpu_pointer;
            }
#endif
            move();
#endif

            return *m_cpu_pointer;
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
                              if (m_pointer_record->m_cpu_destroy) {
                                 m_pointer_record->m_cpu_destroy(voidPointer);
                              }
                              else {
                                 delete pointer;
                              }
                              m_cpu_pointer = nullptr;
                              break;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
                           case GPU:
                           {
                              if (pointer) {
                                 if (m_pointer_record->m_gpu_destroy) {
                                    m_pointer_record->m_gpu_destroy(voidPointer);
                                 }
                                 else {
                                    destroy_on_device(temp);
                                 }
                                 m_gpu_pointer = nullptr;
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
                     using T_non_const = typename std::remove_const<T>::type;
                     T_non_const* temp = const_cast<T_non_const*>(pointer);
                     void* voidPointer = static_cast<void*>(temp);

                     switch (execSpace) {
                        case CPU:
                           if (m_pointer_record->m_cpu_destroy) {
                              m_pointer_record->m_cpu_destroy(voidPointer);
                           }
                           else {
                              delete pointer;
                           }
                           m_cpu_pointer = nullptr;
                           break;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
                        case GPU:
                        {
                           if (pointer) {
                              if (m_pointer_record->m_gpu_destroy) {
                                 m_pointer_record->m_gpu_destroy(voidPointer);
                              }
                              else {
                                 destroy_on_device(pointer);
                              }
                              m_gpu_pointer = nullptr;
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
         T* m_gpu_pointer = nullptr; /// The GPU pointer
#endif
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
#if !defined(CHAI_DISABLE_RM)
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

   /// 
   /// @author Peter Robinson
   ///
   /// A wrapper used by the make_managed family of functions to indicate when
   /// the internal pointers contained by a ManagedArray of managed_ptr should be extracted.
   /// It is not intended to be used directly, but rather created by unpack.
   ///
   template <typename T>
   class ManagedArrayOfManagedPtrUnpacker {
      public:
         CHAI_HOST ManagedArrayOfManagedPtrUnpacker() = delete;

         ///
         /// @author Peter Robinson
         ///
         /// Constructor
         ///
         /// @param[in] arg The ManagedArray of managed_ptr to unpack
         ///
         /// @return a new instance of ManagedArrayOfManagedPtrUnpacker
         ///
         explicit CHAI_HOST ManagedArrayOfManagedPtrUnpacker(const chai::ManagedArray<chai::managed_ptr<T>>& arg)
            : m_array{arg}, m_size(arg.size()), m_ownsData(true)
         {
            // Extract the CPU raw pointers
            m_cpu_ptrs = new T*[m_size];
            for (size_t i = 0; i < m_size; i++) {
               m_cpu_ptrs[i] = m_array[i].get(CPU);
            }
            
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            // Extract the GPU raw pointers
            auto gpu_ptrs = new T*[m_size];
            for (size_t i = 0; i < m_size; i++) {
               gpu_ptrs[i] = m_array[i].get(GPU);
            }
            
            // Allocate and copy device pointers to GPU memory
            gpuMalloc((void**)&m_gpu_ptrs, m_size * sizeof(T*));
            gpuMemcpy(m_gpu_ptrs, gpu_ptrs, m_size * sizeof(T*), gpuMemcpyHostToDevice);
            delete[] gpu_ptrs;
#endif
         }

         /// Copy constructor - copies pointers but doesn't take ownership
         CHAI_HOST ManagedArrayOfManagedPtrUnpacker(const ManagedArrayOfManagedPtrUnpacker& other)
            : m_array(other.m_array),
              m_size(other.m_size),
              m_cpu_ptrs(other.m_cpu_ptrs),
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
              m_gpu_ptrs(other.m_gpu_ptrs),
#endif
              m_ownsData(false) // Copy doesn't own the data
         {
         }
         
         /// Move constructor - takes ownership of resources
         CHAI_HOST ManagedArrayOfManagedPtrUnpacker(ManagedArrayOfManagedPtrUnpacker&& other) noexcept
            : m_array(std::move(other.m_array)),
              m_size(other.m_size),
              m_cpu_ptrs(other.m_cpu_ptrs),
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
              m_gpu_ptrs(other.m_gpu_ptrs),
#endif
              m_ownsData(other.m_ownsData)
         {
            // Transfer ownership
            other.m_ownsData = false;
            other.m_cpu_ptrs = nullptr;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            other.m_gpu_ptrs = nullptr;
#endif
         }
         
         ///
         /// @author Peter Robinson
         ///
         /// Destructor - clean up the arrays of raw pointers only if this owns the data
         ///
         CHAI_HOST ~ManagedArrayOfManagedPtrUnpacker() {
            if (m_ownsData) {
               delete[] m_cpu_ptrs;
            
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               gpuFree(m_gpu_ptrs);
#endif
            }
         }
         
         /// Assignment operator - doesn't take ownership
         CHAI_HOST ManagedArrayOfManagedPtrUnpacker& operator=(const ManagedArrayOfManagedPtrUnpacker& other) {
            if (this != &other) {
               // Clean up own resources if we own them
               if (m_ownsData) {
                  delete[] m_cpu_ptrs;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
                  gpuFree(m_gpu_ptrs);
#endif
               }
               
               // Copy data from other
               m_array = other.m_array;
               m_size = other.m_size;
               m_cpu_ptrs = other.m_cpu_ptrs;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               m_gpu_ptrs = other.m_gpu_ptrs;
#endif
               m_ownsData = false; // Assignment doesn't take ownership
            }
            return *this;
         }
         
         /// Move assignment - takes ownership
         CHAI_HOST ManagedArrayOfManagedPtrUnpacker& operator=(ManagedArrayOfManagedPtrUnpacker&& other) noexcept {
            if (this != &other) {
               // Clean up own resources if we own them
               if (m_ownsData) {
                  delete[] m_cpu_ptrs;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
                  gpuFree(m_gpu_ptrs);
#endif
               }
               
               // Move data from other
               m_array = std::move(other.m_array);
               m_size = other.m_size;
               m_cpu_ptrs = other.m_cpu_ptrs;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               m_gpu_ptrs = other.m_gpu_ptrs;
#endif
               m_ownsData = other.m_ownsData;
               
               // Clear other's ownership
               other.m_ownsData = false;
               other.m_cpu_ptrs = nullptr;
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
               other.m_gpu_ptrs = nullptr;
#endif
            }
            return *this;
         }

         ///
         /// @author Peter Robinson
         ///
         /// Unpacks the data based on execution context
         ///
         /// @return the unpacked data as T**
         ///
         CHAI_HOST_DEVICE T** data() const {
#if defined(CHAI_DEVICE_COMPILE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            return m_gpu_ptrs;
#else
            return m_cpu_ptrs;
#endif
         }

      private:
         chai::ManagedArray<chai::managed_ptr<T>> m_array; //!< The ManagedArray of managed_ptr to unpack
         size_t m_size;
         T** m_cpu_ptrs = nullptr; //!< Array of extracted raw CPU pointers
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
         T** m_gpu_ptrs = nullptr; //!< Device memory array containing GPU pointers
#endif
         bool m_ownsData = false; //!< Flag indicating if this object owns the data and should clean up
   };

   ///
   /// @brief Non-owning host/device view of a persistent table of raw pointers.
   ///
   /// @details PointerTableView is intended for passing a raw-pointer table to
   /// make_managed. It selects the table associated with the calling execution
   /// space, while PointerTable retains the allocations that back both tables.
   ///
   template <typename T>
   class PointerTableView {
      public:
         CHAI_HOST_DEVICE PointerTableView() = default;

         CHAI_HOST_DEVICE PointerTableView(T** cpuPointers, T** gpuPointers)
            : m_cpu_pointers(cpuPointers),
              m_gpu_pointers(gpuPointers)
         {
         }

         CHAI_HOST_DEVICE T** data() const
         {
#if defined(CHAI_DEVICE_COMPILE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            return m_gpu_pointers;
#else
            return m_cpu_pointers;
#endif
         }

      private:
         T** m_cpu_pointers = nullptr;
         T** m_gpu_pointers = nullptr;
   };

   namespace detail {

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      template <typename T>
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      CHAI_HOST void emplace_pointer_table(T** pointers,
                                           const managed_ptr<T>* managedPointers,
                                           size_t size)
      {
         using pointer_type = T*;
         for (size_t index = 0; index < size; ++index) {
            ::new (static_cast<void*>(pointers + index)) pointer_type(managedPointers[index].get(GPU));
         }
      }
#else
      CHAI_GLOBAL void emplace_pointer_table(T** pointers,
                                             const managed_ptr<T>* managedPointers,
                                             size_t size)
      {
         using pointer_type = T*;
         size_t const index = blockIdx.x * blockDim.x + threadIdx.x;
         if (index < size) {
            ::new (static_cast<void*>(pointers + index)) pointer_type(managedPointers[index].get());
         }
      }
#endif
#endif

   } // namespace detail

   ///
   /// @brief Owns Umpire-backed raw-pointer tables for a ManagedArray of managed_ptr.
   ///
   /// @details The host and device tables contain the corresponding raw pointer
   /// from each managed_ptr. The table is deliberately separate from the input
   /// ManagedArray so an object that stores the returned T** can retain it past
   /// the temporary unpacking expression used during construction.
   ///
   template <typename T>
   class PointerTable {
      public:
         CHAI_HOST explicit PointerTable(const chai::ManagedArray<chai::managed_ptr<T>>& managedPointers)
            : m_size(managedPointers.size())
         {
            if (m_size == 0) {
               return;
            }

            auto* arrayManager = chai::ArrayManager::getInstance();
            m_cpu_allocator_id = arrayManager->getAllocatorId(CPU);
            auto cpuAllocator = arrayManager->getAllocator(m_cpu_allocator_id);
            m_cpu_pointers = static_cast<T**>(cpuAllocator.allocate(m_size * sizeof(T*)));

            using pointer_type = T*;
            for (size_t index = 0; index < m_size; ++index) {
               ::new (static_cast<void*>(m_cpu_pointers + index)) pointer_type(managedPointers[index].get(CPU));
            }

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            m_gpu_allocator_id = arrayManager->getAllocatorId(GPU);
            auto gpuAllocator = arrayManager->getAllocator(m_gpu_allocator_id);
            m_gpu_pointers = static_cast<T**>(gpuAllocator.allocate(m_size * sizeof(T*)));

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
            arrayManager->setGPUSimMode(true);
            detail::emplace_pointer_table(m_gpu_pointers, managedPointers.data(GPU), m_size);
            arrayManager->setGPUSimMode(false);
#elif defined(__CUDACC__)
            constexpr int threadsPerBlock = 256;
            int const blocks = static_cast<int>((m_size + threadsPerBlock - 1) / threadsPerBlock);
            detail::emplace_pointer_table<T><<<blocks, threadsPerBlock>>>(m_gpu_pointers,
                                                                            managedPointers.data(GPU),
                                                                            m_size);
#elif defined(__HIPCC__)
            constexpr int threadsPerBlock = 256;
            int const blocks = static_cast<int>((m_size + threadsPerBlock - 1) / threadsPerBlock);
            hipLaunchKernelGGL(detail::emplace_pointer_table<T>, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                               m_gpu_pointers, managedPointers.data(GPU), m_size);
#endif
            synchronize();
#endif
         }

         PointerTable(const PointerTable&) = delete;
         PointerTable& operator=(const PointerTable&) = delete;

         CHAI_HOST ~PointerTable()
         {
            auto* arrayManager = chai::ArrayManager::getInstance();

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
            if (m_gpu_pointers != nullptr) {
               arrayManager->getAllocator(m_gpu_allocator_id).deallocate(m_gpu_pointers);
            }
#endif
            if (m_cpu_pointers != nullptr) {
               arrayManager->getAllocator(m_cpu_allocator_id).deallocate(m_cpu_pointers);
            }
         }

         CHAI_HOST PointerTableView<T> view() const
         {
            return PointerTableView<T>(m_cpu_pointers, m_gpu_pointers);
         }

      private:
         size_t m_size = 0;
         int m_cpu_allocator_id = -1;
         int m_gpu_allocator_id = -1;
         T** m_cpu_pointers = nullptr;
         T** m_gpu_pointers = nullptr;
   };

   ///
   /// @brief Creates and retains a PointerTable while exposing its execution-space view.
   ///
   /// @details Instances are copyable so one can be passed to a managed_ptr
   /// callback. The callback capture then keeps the Umpire-backed tables alive
   /// for exactly the lifetime of the object that stores the view's raw T**.
   ///
   template <typename T>
   class ManagedPtrOfPointerTableUnpacker {
      public:
         CHAI_HOST explicit ManagedPtrOfPointerTableUnpacker(
            const chai::ManagedArray<chai::managed_ptr<T>>& managedPointers)
            : m_table(std::make_shared<PointerTable<T>>(managedPointers))
         {
         }

         CHAI_HOST PointerTableView<T> view() const
         {
            return m_table->view();
         }

      private:
         std::shared_ptr<PointerTable<T>> m_table;
   };

   namespace detail {

      using destroyer_type = std::function<void(void*)>;

      CHAI_HOST inline std::unordered_map<void*, destroyer_type>& destroyer_map()
      {
         static std::unordered_map<void*, destroyer_type> s_destroyers;
         return s_destroyers;
      }

      CHAI_HOST inline std::mutex& destroyer_mutex()
      {
         static std::mutex s_mutex;
         return s_mutex;
      }

      CHAI_HOST inline void register_destroyer(void* pointer, destroyer_type destroyer)
      {
         std::lock_guard<std::mutex> lock(destroyer_mutex());
         destroyer_map()[pointer] = std::move(destroyer);
      }

      CHAI_HOST inline destroyer_type release_destroyer(void* pointer)
      {
         std::lock_guard<std::mutex> lock(destroyer_mutex());
         auto& destroyers = destroyer_map();
         auto itr = destroyers.find(pointer);
         if (itr == destroyers.end()) {
            return {};
         }

         destroyer_type destroyer = std::move(itr->second);
         destroyers.erase(itr);
         return destroyer;
      }

      template <typename T>
      CHAI_HOST T* allocate_from_space(ExecutionSpace space)
      {
         auto allocator = ArrayManager::getInstance()->getAllocator(space);
         return static_cast<T*>(allocator.allocate(sizeof(T)));
      }

      template <typename T>
      CHAI_HOST bool is_umpire_allocation(T* pointer)
      {
         return pointer != nullptr &&
                umpire::ResourceManager::getInstance().hasAllocator(static_cast<void*>(pointer));
      }

      template <typename T>
      CHAI_HOST void deallocate_umpire_allocation(T* pointer)
      {
         if (pointer != nullptr) {
            auto& resourceManager = umpire::ResourceManager::getInstance();
            auto allocator = resourceManager.getAllocator(static_cast<void*>(pointer));
            allocator.deallocate(static_cast<void*>(pointer));
         }
      }

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

      ///
      /// @author Peter Robinson
      ///
      /// This implementation of processArguments handles the CHAI ManagedArrayOfManagedPtrUnpacker type.
      ///
      /// @param[in] arg The ManagedArrayOfManagedPtrUnpacker from which to extract a raw pointer array
      ///
      /// @return raw pointer array extracted from the managed_ptr array
      ///
      template <typename T>
      CHAI_HOST_DEVICE T** processArguments(const ManagedArrayOfManagedPtrUnpacker<T>& arg) {
         return arg.data();
      }

      ///
      /// @brief Extracts the execution-space pointer table from a PointerTableView.
      ///
      template <typename T>
      CHAI_HOST_DEVICE T** processArguments(const PointerTableView<T>& arg) {
         return arg.data();
      }

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)

      template <typename T,
                typename... Args>
      CHAI_GLOBAL void emplace_on_device(T* gpuPointer, Args... args)
      {
         ::new (static_cast<void*>(gpuPointer)) T(processArguments(args)...);
      }

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
      CHAI_GLOBAL void make_on_device(T** gpuPointer, Args... args)
      {
         *gpuPointer = new T(processArguments(args)...);
      }

      template <typename T>
      CHAI_GLOBAL void destroy_in_place_on_device(T* gpuPointer)
      {
         if (gpuPointer != nullptr) {
            gpuPointer->~T();
         }
      }

      ///
      /// @author Alan Dayton
      ///
      /// Destroys the device pointer.
      ///
      /// @param[out] gpuPointer The device pointer to call delete on
      ///
      template <typename T>
      CHAI_GLOBAL void destroy_on_device(T* gpuPointer)
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
/// @author User
///
/// Unpacks the pointers contained in the ManagedArray of managed_ptr and passes them
/// as a T** array.
///
/// @param[in] arg The ManagedArray of managed_ptr to unpack
///
/// @return A wrapper used by make_managed for unpacking the internal pointers
///         in the correct space
///
template <typename T>
CHAI_HOST ManagedArrayOfManagedPtrUnpacker<T> unpack(const chai::ManagedArray<chai::managed_ptr<T>>& arg) {
   return ManagedArrayOfManagedPtrUnpacker<T>(arg);
}

///
/// @brief Creates a persistent Umpire-backed pointer table for managed pointers.
///
/// @details Capture the returned object in the callback of any managed_ptr that
/// stores its view, so its table remains valid until that managed_ptr is freed.
///
template <typename T>
CHAI_HOST ManagedPtrOfPointerTableUnpacker<T> unpack_pointer_table(
   const chai::ManagedArray<chai::managed_ptr<T>>& arg)
{
   return ManagedPtrOfPointerTableUnpacker<T>(arg);
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
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
#if !defined(CHAI_DISABLE_RM)
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();

      // Set the execution space so that ManagedArrays and managed_ptrs
      // are handled properly
      arrayManager->setExecutionSpace(CPU);
#endif

      T* cpuPointer = detail::allocate_from_space<T>(CPU);

      ::new (static_cast<void*>(cpuPointer)) T(detail::processArguments(args)...);

      detail::register_destroyer(static_cast<void*>(cpuPointer), [] (void* pointer) {
         static_cast<T*>(pointer)->~T();
      });

#if !defined(CHAI_DISABLE_RM)
      // Set the execution space back to the previous value
      arrayManager->setExecutionSpace(currentSpace);
#endif

      // Return the CPU pointer
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
      if (detail::is_umpire_allocation(cpuPointer)) {
         auto destroyer = detail::release_destroyer(static_cast<void*>(cpuPointer));
         if (destroyer) {
            destroyer(static_cast<void*>(cpuPointer));
         }
         else {
            cpuPointer->~T();
         }
         detail::deallocate_umpire_allocation(cpuPointer);
      }
      else {
         delete cpuPointer;
      }
   }

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)

   ///
   /// @author Alan Dayton
   ///
   /// Creates a new T on the device.
   ///
   /// @param[in] args The arguments to T's constructor
   ///
   /// @return The device pointer to the new T
   ///
   template <typename T,
             typename... Args>
   CHAI_HOST T* make_on_device(Args... args) {
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
#if !defined(CHAI_DISABLE_RM)
      ExecutionSpace currentSpace = arrayManager->getExecutionSpace();
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      arrayManager->setGPUSimMode(true);
#endif

      // Set the execution space so that ManagedArrays and managed_ptrs
      // are handled properly
      arrayManager->setExecutionSpace(GPU);
#endif

      T* gpuPointer = detail::allocate_from_space<T>(GPU);

      // Create the object on the device
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      detail::emplace_on_device(gpuPointer, args...);
      arrayManager->setGPUSimMode(false);
#elif defined(__CUDACC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      detail::emplace_on_device<<<1, 1>>>(gpuPointer, args...);
#elif defined(__HIPCC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      hipLaunchKernelGGL(detail::emplace_on_device, 1, 1, 0, 0, gpuPointer, args...);
#endif

      detail::register_destroyer(static_cast<void*>(gpuPointer), [] (void* pointer) {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
         chai::ArrayManager* destroyArrayManager = chai::ArrayManager::getInstance();
         destroyArrayManager->setGPUSimMode(true);
         detail::destroy_in_place_on_device(static_cast<T*>(pointer));
         destroyArrayManager->setGPUSimMode(false);
#elif defined(__CUDACC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
         detail::destroy_in_place_on_device<<<1, 1>>>(static_cast<T*>(pointer));
#elif defined(__HIPCC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
         hipLaunchKernelGGL(detail::destroy_in_place_on_device, 1, 1, 0, 0, static_cast<T*>(pointer));
#endif
      });

#if !defined(CHAI_DISABLE_RM)
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
      if (gpuPointer == nullptr) {
         return;
      }

      bool const isUmpireAllocation = detail::is_umpire_allocation(gpuPointer);
      auto destroyer = detail::release_destroyer(static_cast<void*>(gpuPointer));

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      arrayManager->setGPUSimMode(true);
      if (isUmpireAllocation) {
         if (destroyer) {
            destroyer(static_cast<void*>(gpuPointer));
         }
         else {
            detail::destroy_in_place_on_device(gpuPointer);
         }
      }
      else {
         detail::destroy_on_device(gpuPointer);
      }
      arrayManager->setGPUSimMode(false);
#elif defined(__CUDACC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      if (isUmpireAllocation) {
         if (destroyer) {
            destroyer(static_cast<void*>(gpuPointer));
         }
         else {
            detail::destroy_in_place_on_device<<<1, 1>>>(gpuPointer);
         }
      }
      else {
         detail::destroy_on_device<<<1, 1>>>(gpuPointer);
      }
#elif defined(__HIPCC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      if (isUmpireAllocation) {
         if (destroyer) {
            destroyer(static_cast<void*>(gpuPointer));
         }
         else {
            hipLaunchKernelGGL(detail::destroy_in_place_on_device, 1, 1, 0, 0, gpuPointer);
         }
      }
      else {
         hipLaunchKernelGGL(detail::destroy_on_device, 1, 1, 0, 0, gpuPointer);
      }
#endif

      if (isUmpireAllocation) {
         synchronize();
         detail::deallocate_umpire_allocation(gpuPointer);
      }
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      // Construct on the GPU first to take advantage of asynchrony
      T* gpuPointer = make_on_device<T>(args...);

      // Host construction may consume arguments initialized asynchronously by
      // device construction. Complete that work before callers can release
      // the pooled allocations backing those arguments.
      synchronize();
#endif

      // Construct on the CPU
      T* cpuPointer = make_on_host<T>(args...);

      // Construct the managed_ptr and retain concrete destroyers so converted
      // managed_ptr<Base> instances still destroy the most-derived object.
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      managed_ptr<T> result({CPU, GPU}, {cpuPointer, gpuPointer});
      result.m_pointer_record->m_gpu_destroy = [] (void* pointer) {
         destroy_on_device(static_cast<T*>(pointer));
      };
#else
      managed_ptr<T> result({CPU}, {cpuPointer});
#endif
      result.m_pointer_record->m_cpu_destroy = [] (void* pointer) {
         destroy_on_host(static_cast<T*>(pointer));
      };
      return result;
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

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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

#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
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
#if (defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      std::swap(lhs.m_gpu_pointer, rhs.m_gpu_pointer);
#endif
      std::swap(lhs.m_pointer_record, rhs.m_pointer_record);
   }
} // namespace chai

#else // defined(CHAI_ENABLE_MANAGED_PTR)

#error CHAI must be configured with -DCHAI_ENABLE_MANAGED_PTR=ON to use managed_ptr! \
       If CHAI_ENABLE_MANAGED_PTR is defined as a macro, it is safe to include managed_ptr.hpp.

#endif // defined(CHAI_ENABLE_MANAGED_PTR)

#endif // MANAGED_PTR
