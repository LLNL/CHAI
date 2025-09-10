//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_ARRAY_MANAGER_HPP
#define CHAI_ARRAY_MANAGER_HPP

#include "chai/expt/Context.hpp"
#include <cstddef>

namespace chai {
namespace expt {
  /*!
   * \class ArrayManager
   *
   * \brief Controls the coherence of an array.
   */
  class ArrayManager {
    public:
      /*!
       * \brief Virtual destructor.
       */
      virtual ~ArrayManager() = default;

      /*!
       * \brief Creates a clone of this ArrayManager.
       *
       * \return A new ArrayManager object that is a clone of this instance.
       */
      virtual ArrayManager* clone() const = 0;

      /*!
       * \brief Resizes the array to the specified new size.
       *
       * \param newSize The new size to resize the array to.
       */
      virtual void resize(std::size_t newSize) = 0;

      /*!
       * \brief Returns the size of the contained array.
       *
       * \return The size of the contained array.
       */
      virtual std::size_t size() const = 0;

      /*!
       * \brief Updates the data to be coherent in the current execution context.
       *
       * \param data [out] A coherent array in the current execution context.
       */
      virtual T* data(Context context, bool touch) = 0;

      /*!
       * \brief Returns the value at index i.
       *
       * Note: Use this function sparingly as it may be slow.
       *
       * \param i The index of the element to get.
       * \return The value at index i.
       */
      virtual T get(std::size_t i) const = 0;

      /*!
       * \brief Sets the value at index i to the specified value.
       *
       * Note: Use this function sparingly as it may be slow.
       *
       * \param i The index of the element to set.
       * \param value The value to set at index i.
       */
      virtual void set(std::size_t i, const T& value) = 0;
  };  // class ArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_MANAGER_HPP
