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

#include "chai/ChaiMacros.hpp"

// Standard libary headers
#include <cstddef>


namespace chai {
   ///
   /// @class managed_ptr<T>
   /// @author Alan Dayton
   ///
   template <class T, class TPtr = T*, class TRef = T&>
   class managed_ptr {
      public:
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
         /// Constructs a managed_ptr from the given pointer.
         ///    U* must be convertible to T*.
         ///
         /// @param[in] ptr A pointer
         ///
         template <class UPtr>
         explicit managed_ptr(UPtr&& ptr) : m_ptr(std::forward<UPtr>(ptr))
         {
            static_assert(std::is_convertible<UPtr, TPtr>::value,
                          "UPtr must be convertible to TPtr.");
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy constructor.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         managed_ptr(const managed_ptr& other) noexcept = default;

         ///
         /// @author Alan Dayton
         ///
         /// Converting constructor.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template <class U, class UPtr = U*, class URef = U&>
         CHAI_HOST_DEVICE managed_ptr(const managed_ptr<U, UPtr, URef>& other) noexcept :
            m_ptr(other.m_ptr)
         {
            static_assert(std::is_convertible<UPtr, TPtr>::value,
                          "UPtr must be convertible to TPtr.");
         }

         ///
         /// @author Alan Dayton
         ///
         /// Copy assignment operator.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         managed_ptr& operator=(const managed_ptr& other) noexcept = default;

         ///
         /// @author Alan Dayton
         ///
         /// Conversion copy assignment operator.
         ///
         /// @param[in] other The managed_ptr to copy
         ///
         template <class U, class UPtr = U*, class URef = U&>
         CHAI_HOST_DEVICE managed_ptr& operator=(const managed_ptr<U, UPtr, URef>& other) noexcept {
            static_assert(std::is_convertible<UPtr, TPtr>::value,
                          "UPtr must be convertible to TPtr.");

            m_ptr = other.m_ptr;

            return *this;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         /// TODO: Should this be a variadic template and forward arguments to the getter?
         ///
         CHAI_HOST_DEVICE inline TPtr get() const {
            return m_ptr;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU pointer depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline TPtr operator->() const {
            return m_ptr;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns the CPU or GPU reference depending on the calling context.
         ///
         CHAI_HOST_DEVICE inline TRef operator*() const {
            return *m_ptr;
         }

         ///
         /// @author Alan Dayton
         ///
         /// Returns true if the contained pointer is not nullptr, false otherwise.
         ///
         CHAI_HOST_DEVICE inline explicit operator bool() const noexcept {
            return get() != nullptr;
         }

      private:
         TPtr m_ptr = nullptr; /// The pointer

         /// Needed for the converting constructor
         template <class U, class UPtr = U*, class URef = U&>
         friend class managed_ptr;

         /// Needed to use the make_managed API
         template <class U, class UPtr = U*, class URef = U&,
                   class... Args>
         friend CHAI_HOST managed_ptr<U, UPtr, URef> make_managed(Args... args);
   };

   ///
   /// @author Alan Dayton
   ///
   /// Makes a managed_ptr<T>.
   /// Factory function to create managed_ptrs.
   ///
   /// @param[in] args The arguments to T's constructor
   ///
   template <class T, class TPtr = T*, class TRef = T&,
             class... Args>
   CHAI_HOST managed_ptr<T, TPtr, TRef> make_managed(Args&&... args) {
      return managed_ptr<T, TPtr, TRef>(make_managed<TPtr>(std::forward<Args>(args)...));
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
   template <class T, class TPtr = T*, class TRef = T&,
             class U = T, class UPtr = U*, class URef = U&>
   CHAI_HOST managed_ptr<T, TPtr, TRef> static_pointer_cast(const managed_ptr<U, UPtr, URef>& other) noexcept {
      return managed_ptr<T, TPtr, TRef>(static_pointer_cast<TPtr>(other.get()));
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
   template <class T, class TPtr = T*, class TRef = T&,
             class U = T, class UPtr = U*, class URef = U&>
   CHAI_HOST managed_ptr<T, TPtr, TRef> dynamic_pointer_cast(const managed_ptr<U, UPtr, URef>& other) noexcept {
      return managed_ptr<T, TPtr, TRef>(dynamic_pointer_cast<TPtr>(other.get()));
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
   template <class T, class TPtr = T*, class TRef = T&,
             class U = T, class UPtr = U*, class URef = U&>
   CHAI_HOST managed_ptr<T, TPtr, TRef> const_pointer_cast(const managed_ptr<U, UPtr, URef>& other) noexcept {
      return managed_ptr<T, TPtr, TRef>(const_pointer_cast<TPtr>(other.get()));
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
   template <class T, class TPtr = T*, class TRef = T&,
             class U = T, class UPtr = U*, class URef = U&>
   CHAI_HOST managed_ptr<T, TPtr, TRef> reinterpret_pointer_cast(const managed_ptr<U, UPtr, URef>& other) noexcept {
      return managed_ptr<T, TPtr, TRef>(reinterpret_pointer_cast<TPtr>(other.get()));
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
   template <class T, class TPtr = T*, class TRef = T&,
             class U = T, class UPtr = U*, class URef = U&>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T, TPtr, TRef>& lhs,
                   const managed_ptr<U, UPtr, URef>& rhs) noexcept {
      return lhs.get() == rhs.get();
   }

   /// Comparison operators with nullptr

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] lhs The managed_ptr to compare to nullptr
   ///
   template <class T, class TPtr = T*, class TRef = T&>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(const managed_ptr<T, TPtr, TRef>& lhs, std::nullptr_t) noexcept {
      return lhs.get() == nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] rhs The managed_ptr to compare to nullptr
   ///
   template <class T, class TPtr = T*, class TRef = T&>
   CHAI_HOST_DEVICE CHAI_INLINE
   bool operator==(std::nullptr_t, const managed_ptr<T, TPtr, TRef>& rhs) noexcept {
      return nullptr == rhs.get();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison.
   ///
   /// @param[in] lhs The first managed_ptr to swap
   /// @param[in] rhs The second managed_ptr to swap
   ///
   template <class T, class TPtr = T*, class TRef = T&>
   void swap(managed_ptr<T, TPtr, TRef>& lhs, managed_ptr<T, TPtr, TRef>& rhs) noexcept {
      std::swap(lhs.m_ptr, rhs.m_ptr);
   }
} // namespace chai

#else // defined(CHAI_ENABLE_MANAGED_PTR)

#error CHAI must be configured with -DCHAI_ENABLE_MANAGED_PTR=ON to use managed_ptr! \
       If CHAI_ENABLE_MANAGED_PTR is defined as a macro, it is safe to include managed_ptr.hpp.

#endif // defined(CHAI_ENABLE_MANAGED_PTR)

#endif // MANAGED_PTR

