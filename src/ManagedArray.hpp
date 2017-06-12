#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/ChaiMacros.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/Types.hpp"

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

  /*!
   * \brief Cast the ManagedArray to a raw pointer.
   *
   * \return Raw pointer to data.
   */
  CHAI_HOST_DEVICE operator T*() const;

  /*!
   * \brief 
   *
   */
  template<bool B = std::is_const<T>::value,typename std::enable_if<!B, int>::type = 0>
  CHAI_HOST_DEVICE operator ManagedArray<const T> () const;

  CHAI_HOST_DEVICE ManagedArray(T* data, ArrayManager* array_manager, uint m_elems);

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

#include "chai/ManagedArray.inl"

#endif // CHAI_ManagedArray_HPP
