//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_HOST_SHARED_POINTER_HPP
#define CHAI_HOST_SHARED_POINTER_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include <cstddef>
#include <memory>
#include <new>

namespace chai::expt
{
  /*!
   * \brief HostSharedPointer is a wrapper around std::shared_ptr that acts
   *        like std::shared_ptr on the host but has no effect on the device.
   *
   * \tparam T The type of object contained.
   */
  template <typename T>
  class HostSharedPointer
  {
    private:
      using SharedPointer = std::shared_ptr<T>;

    public:
      constexpr HostSharedPointer() noexcept
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer();
      }

      constexpr HostSharedPointer(std::nullptr_t) noexcept
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(nullptr);
      }

      template <typename U>
      explicit HostSharedPointer(U* ptr)
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(ptr);
      }

      template <typename U>
      HostSharedPointer(const HostSharedPointer<U>& other) noexcept
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(other);
      }

      CHAI_HOST_DEVICE HostSharedPointer(const HostSharedPointer& other) noexcept
      {
#if !defined(CHAI_DEVICE_COMPILE)
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(other.shared_ptr());
#endif
      }

      HostSharedPointer(HostSharedPointer&& other) noexcept
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(std::move(other.shared_ptr()));
      }

      template <typename U>
      HostSharedPointer(HostSharedPointer<U>&& other) noexcept
      {
        ::new (static_cast<void*>(std::addressof(m_storage))) SharedPointer(std::move(other.shared_ptr()));
      }

      CHAI_HOST_DEVICE ~HostSharedPointer()
      {
#if !defined(CHAI_DEVICE_COMPILE)
        shared_ptr().~SharedPointer();
#endif
      }

      HostSharedPointer& operator=(const HostSharedPointer& other) noexcept
      {
        shared_ptr() = other.shared_ptr();
        return *this;
      }

      template <typename U>
      HostSharedPointer& operator=(const HostSharedPointer<U>& other) noexcept
      {
        shared_ptr() = other.shared_ptr();
        return *this;
      }

      HostSharedPointer& operator=(HostSharedPointer&& other) noexcept
      {
        shared_ptr() = other.shared_ptr();
        return *this;
      }

      template <typename U>
      HostSharedPointer& operator=(HostSharedPointer<U>&& other) noexcept
      {
        shared_ptr() = other.shared_ptr();
        return *this;
      }

      T* get() const noexcept
      {
        return shared_ptr().get();
      }

      T& operator*() const noexcept
      {
        return *get();
      }

      T* operator->() const noexcept
      {
        return get();
      }

      explicit operator bool() const noexcept
      {
        return get() != nullptr;
      }

    private:
      /*!
       * \brief Raw, properly-aligned storage for the underlying std::shared_ptr<T>.
       *
       * The shared_ptr is constructed/destructed in-place via placement new to avoid
       * device-side effects when copy constructed or destroyed on the device.
       */
      alignas(SharedPointer) std::byte m_storage[sizeof(SharedPointer)];

      /*!
       * \brief Access the underlying std::shared_ptr<T> instance.
       *
       * \note The std::shared_ptr<T> is stored in raw aligned storage and constructed
       *       in-place via placement new. std::launder is used to obtain a valid
       *       pointer to the active object.
       *
       * \return A mutable reference to the contained std::shared_ptr<T>.
       */
      SharedPointer& shared_ptr()
      {
        return *std::launder(reinterpret_cast<SharedPointer*>(std::addressof(m_storage)));
      }

      /*!
       * \brief Access the underlying std::shared_ptr<T> instance (const).
       *
       * \note The std::shared_ptr<T> is stored in raw aligned storage and constructed
       *       in-place via placement new. std::launder is used to obtain a valid
       *       pointer to the active object.
       *
       * \return A const reference to the contained std::shared_ptr<T>.
       */
      const SharedPointer& shared_ptr() const
      {
        return *std::launder(reinterpret_cast<const SharedPointer*>(std::addressof(m_storage)));
      }
  };  // class HostSharedPointer
}  // namespace chai::expt

#endif  // CHAI_HOST_SHARED_POINTER_HPP
