// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
// rights reserved.
// 
// Produced at the Lawrence Livermore National Laboratory.
// 
// This file is part of CHAI.
// 
// LLNL-CODE-705877
// 
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
// 
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/config.hpp"

#include "chai/ChaiMacros.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/Types.hpp"


#include <cstddef>

namespace chai {

/*!
 * \class ManagedArray
 *
 * \brief Provides an array-like class that automatically transfers data
 * between memory spaces.
 *
 * The ManagedArray class interacts with the ArrayManager to provide a
 * smart-array object that will automatically move its data between different
 * memory spaces on the system. Data motion happens when the copy constructor
 * is called, so the ManagedArray works best when used in co-operation with a
 * programming model like RAJA.
 *
 * \include ./examples/ex1.cpp
 *
 * \tparam T The type of elements stored in the ManagedArray.
 */
template <typename T>
class ManagedArray {
  public:

  using T_non_const = typename std::remove_const<T>::type;

  /*!
   * \brief Default constructor creates a ManagedArray with no allocations.
   */
  CHAI_HOST_DEVICE ManagedArray();

  /*!
   * \brief Constructor to create a ManagedArray with specified size, allocated
   * in the provided space.
   *
   * The default space for allocations is CPU.
   *
   * \param elems Number of elements in the array.
   * \param space Execution space in which to allocate the array.
   */
  CHAI_HOST_DEVICE ManagedArray(uint elems, ExecutionSpace space=CPU);

  /*!
   * \brief Copy constructor handles data movement.
   *
   * The copy constructor interacts with the ArrayManager to move the
   * ManagedArray's data between execution spaces.
   *
   * \param other ManagedArray being copied.
   */
  CHAI_HOST_DEVICE ManagedArray(ManagedArray const& other);

  /*!
   * \brief Construct a ManagedArray from a nullptr.
   */
  CHAI_HOST_DEVICE ManagedArray(std::nullptr_t other);

  /*!
   * \brief Allocate data for the ManagedArray in the specified space.
   *
   * The default space for allocations is the CPU.
   *
   * \param elems Number of elements to allocate.
   * \param space Execution space in which to allocate data.
   */
  CHAI_HOST void allocate(uint elems, ExecutionSpace space=CPU);

  /*!
   * \brief Reallocate data for the ManagedArray.
   *
   * Reallocation will happen in all spaces the data exists
   *
   * \param elems Number of elements to allocate.
   */
  CHAI_HOST void reallocate(uint elems);

  /*!
   * \brief Free all data allocated by this ManagedArray.
   */
  CHAI_HOST void free();

  /*!
   * \brief Get the number of elements in the array.
   *
   * \return The number of elements in the array
   */
  CHAI_HOST uint size() const;

  /*!
   * \brief Register this ManagedArray object as 'touched' in the given space.
   *
   * \param space The space to register a touch.
   */
  CHAI_HOST void registerTouch(ExecutionSpace space);

  /*!
   * \brief Return reference to i-th element of the ManagedArray.
   *
   * \param i Element to return reference to.
   *
   * \return Reference to i-th element.
   */
  CHAI_HOST_DEVICE T& operator[](const int i) const;

  /*!
   * \brief Set val to the value of element i in the ManagedArray.
   *
   */
  // CHAI_HOST_DEVICE void pick(size_t i, T_non_const& val);

#if defined(ENABLE_IMPLICIT_CONVERSIONS)
  /*!
   * \brief Cast the ManagedArray to a raw pointer.
   *
   * \return Raw pointer to data.
   */
  CHAI_HOST_DEVICE operator T*() const;

  /*!
   * \brief Construct a ManagedArray from a raw pointer.
   *
   * This raw pointer *must* have taken from an existing ManagedArray object.
   *
   * \param data Raw pointer to data.
   * \param enable Boolean argument (unused) added to differentiate constructor.
   */
  template<bool Q=0>
  CHAI_HOST_DEVICE ManagedArray(T* data, bool test=Q);
#endif

  /*!
   * \brief 
   *
   */
  template<bool B = std::is_const<T>::value,typename std::enable_if<!B, int>::type = 0>
  CHAI_HOST_DEVICE operator ManagedArray<const T> () const;

  CHAI_HOST_DEVICE ManagedArray(T* data, ArrayManager* array_manager, uint m_elems);


  CHAI_HOST_DEVICE ManagedArray<T>& operator= (std::nullptr_t);

  private:

  /*! 
   * Currently active data pointer.
   */
  mutable T* m_active_pointer;

  /*! 
   * Pointer to ArrayManager instance.
   */
  ArrayManager* m_resource_manager;

  /*!
   * Number of elements in the ManagedArray.
   */
  uint m_elems;
};

} // end of namespace chai

#if defined(DISABLE_RM)
#include "chai/ManagedArray_thin.inl"
#else
#include "chai/ManagedArray.inl"
#endif

#endif // CHAI_ManagedArray_HPP
