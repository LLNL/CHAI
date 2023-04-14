//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArray_HPP
#define CHAI_ManagedArray_HPP

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/Types.hpp"

#include "umpire/Allocator.hpp"

#include <cstddef>

namespace chai
{

namespace {
inline ExecutionSpace get_default_space() {
  return ArrayManager::getInstance()->getDefaultAllocationSpace();
}

}


struct InvalidConstCast;

/*!
 * \class CHAICopyable
 *
 * \brief Provides an interface for types that are copyable.
 *
 * If a class inherits from CHAICopyable, then the stored items of type T
 * are moved when the copy constructor is called.
 */
class CHAICopyable
{
};

/*!
 * \class CHAIDISAMBIGUATE
 *
 * \brief Type to disambiguate otherwise ambiguous constructors.
 *
 */
class CHAIDISAMBIGUATE
{
public:
  CHAI_HOST_DEVICE CHAIDISAMBIGUATE(){};
  CHAI_HOST_DEVICE ~CHAIDISAMBIGUATE(){};
};
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
class ManagedArray : public CHAICopyable
{
public:
  using T_non_const = typename std::remove_const<T>::type;

  CHAI_HOST_DEVICE ManagedArray();

  /*!
   * \brief Default constructor creates a ManagedArray with no allocations.
   */
  CHAI_HOST_DEVICE ManagedArray(
      std::initializer_list<chai::ExecutionSpace> spaces,
      std::initializer_list<umpire::Allocator> allocators);

  /*!
   * \brief Constructor to create a ManagedArray with specified size, allocated
   * in the provided space.
   *
   * If space is NONE, the storage will be allocated in the default space. The
   * default space for these allocations can be set with the
   * setDefaultAllocationSpace method of the ArrayManager.
   *
   * \param elems Number of elements in the array.
   * \param space Execution space in which to allocate the array.
   */
  CHAI_HOST_DEVICE ManagedArray(size_t elems, ExecutionSpace space = get_default_space());

  CHAI_HOST_DEVICE ManagedArray(
      size_t elems,
      std::initializer_list<chai::ExecutionSpace> spaces,
      std::initializer_list<umpire::Allocator> allocators,
      ExecutionSpace space = NONE);

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

  CHAI_HOST ManagedArray(PointerRecord* record, ExecutionSpace space);

  /*!
   * \brief Allocate data for the ManagedArray in the specified space.
   *
   * The default space for allocations is the CPU.
   *
   * \param elems Number of elements to allocate.
   * \param space Execution space in which to allocate data.
   * \param cback User defined callback for memory events (alloc, free, move)
   */
  CHAI_HOST void allocate(size_t elems,
                          ExecutionSpace space = CPU,
                          UserCallback const& cback =
                          [] (const PointerRecord*, Action, ExecutionSpace) {});



  /*!
   * \brief Reallocate data for the ManagedArray.
   *
   * Reallocation will happen in all spaces the data exists
   *
   * \param elems Number of elements to allocate.
   */
  CHAI_HOST void reallocate(size_t elems);

  /*!
   * \brief Free all data allocated by this ManagedArray.
   */
  CHAI_HOST void free(ExecutionSpace space = NONE);

  /*!
   * \brief Reset array state.
   *
   * The next space that accesses this array will be considered a first touch,
   * and no data will be migrated.
   */
  CHAI_HOST void reset();

  /*!
   * \brief Get the number of elements in the array.
   *
   * \return The number of elements in the array
   */
  CHAI_HOST_DEVICE size_t size() const;

  /*!
   * \brief Register this ManagedArray object as 'touched' in the given space.
   *
   * \param space The space to register a touch.
   */
  CHAI_HOST void registerTouch(ExecutionSpace space);

  //CHAI_HOST void move(ExecutionSpace space=NONE) const;
  CHAI_HOST void move(ExecutionSpace space, camp::resources::Resource* resource);
  CHAI_HOST void move(ExecutionSpace space=NONE,
                      bool registerTouch=!std::is_const<T>::value) const;

  CHAI_HOST_DEVICE ManagedArray<T> slice(size_t begin, size_t elems=(size_t)-1) const;

  /*!
   * \brief Return reference to i-th element of the ManagedArray.
   *
   * \param i Element to return reference to.
   *
   * \return Reference to i-th element.
   */
  template <typename Idx>
  CHAI_HOST_DEVICE T& operator[](const Idx i) const;

  /*!
   * \brief get access to m_active_pointer
   * @return a copy of m_active_base_pointer
   */
  CHAI_HOST_DEVICE T* getActiveBasePointer() const;

  /*!
   * \brief get access to m_active_pointer
   * @return a copy of m_active_pointer
   */
  CHAI_HOST_DEVICE T* getActivePointer() const;

  /*!
   * \brief Move data to the current execution space (actually determined
   *        by where the code is executing) and return a raw pointer.
   *
   * \return Raw pointer to data in the current execution space
   */
  CHAI_HOST_DEVICE T* data() const;

  /*!
   * \brief Move data to the current execution space (actually determined
   *        by where the code is executing) and return a raw pointer. Do
   *        not mark data as touched since a pointer to const is returned.
   *
   * \return Raw pointer to data in the current execution space
   */
  CHAI_HOST_DEVICE const T* cdata() const;

  /*!
   * \brief Return the raw pointer to the data in the given execution
   *        space. Optionally move the data to that execution space.
   *
   * \param space The execution space from which to retrieve the raw pointer.
   * \param do_move Ensure data at that pointer is live and valid.
   *
   * @return A copy of the pointer in the given execution space
   */
  CHAI_HOST T* data(ExecutionSpace space, bool do_move = true) const;

  /*!
   * \brief Deprecated! Use the data method instead!
   *        Return the raw pointer to the data in the given execution
   *        space. Optionally move the data to that execution space.
   *
   * \param space The execution space from which to retrieve the raw pointer.
   * \param do_move Ensure data at that pointer is live and valid.
   *
   * @return A copy of the pointer in the given execution space
   */
  CHAI_HOST T* getPointer(ExecutionSpace space, bool do_move = true) const;

  /*!
   * \brief Move data to the current execution space (actually determined
   *        by where the code is executing) and return an iterator to the
   *        beginning of the array.
   *
   * \return Iterator (as raw pointer) to the start of the array in the
   *         current execution space
   */
  CHAI_HOST_DEVICE T* begin() const;

  /*!
   * \brief Move data to the current execution space (actually determined
   *        by where the code is executing) and return an iterator to
   *        one past the end of the array.
   *
   * \return Iterator (as raw pointer) to the element after the last element
   *         of the array in the current execution space
   */
  CHAI_HOST_DEVICE T* end() const;

  /*!
   * \brief
   *
   */
  //  operator ManagedArray<typename std::conditional<!std::is_const<T>::value,
  //  const T, InvalidConstCast>::type> () const;
  template <typename U = T>
  operator typename std::enable_if<!std::is_const<U>::value,
                                   ManagedArray<const U> >::type() const;


  CHAI_HOST_DEVICE ManagedArray(T* data,
                                ArrayManager* array_manager,
                                size_t m_elems,
                                PointerRecord* pointer_record);

  ManagedArray<T>& operator=(ManagedArray const & other) = default;

  CHAI_HOST_DEVICE ManagedArray<T>& operator=(ManagedArray && other);

  CHAI_HOST_DEVICE ManagedArray<T>& operator=(std::nullptr_t);


  CHAI_HOST_DEVICE bool operator==(const ManagedArray<T>& rhs) const;
  CHAI_HOST_DEVICE bool operator!=(const ManagedArray<T>& from) const;

  CHAI_HOST_DEVICE bool operator==(const T* from) const;
  CHAI_HOST_DEVICE bool operator!=(const T* from) const;

  CHAI_HOST_DEVICE bool operator==(std::nullptr_t from) const;
  CHAI_HOST_DEVICE bool operator!=(std::nullptr_t from) const;


  CHAI_HOST_DEVICE explicit operator bool() const;


#if defined(CHAI_ENABLE_PICK)
  /*!
   * \brief Return the value of element i in the ManagedArray.
   * ExecutionSpace space to the current one
   *
   * \param index The index of the element to be fetched
   * \param space The index of the element to be fetched
   * \return The value of the i-th element in the ManagedArray.
   * \tparam T_non_const The (non-const) type of data value in ManagedArray.
   */
  CHAI_HOST_DEVICE T_non_const pick(size_t i) const;

  /*!
   * \brief Set the value of element i in the ManagedArray to be val.
   *
   * \param index The index of the element to be set
   * \param val Source location of the value
   * \tparam T The type of data value in ManagedArray.
   */
  CHAI_HOST_DEVICE void set(size_t i, T val) const;

  /*!
   * \brief Increment the value of element i in the ManagedArray.
   *
   * \param index The index of the element to be incremented
   * \tparam T The type of data value in ManagedArray.
   */
  CHAI_HOST_DEVICE void incr(size_t i) const;

  /*!
   * \brief Decrement the value of element i in the ManagedArray.
   *
   * \param index The index of the element to be decremented
   * \tparam T The type of data value in ManagedArray.
   */
  CHAI_HOST_DEVICE void decr(size_t i) const;
#endif


#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
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
  template <bool Q = false>
  CHAI_HOST_DEVICE ManagedArray(T* data,
                                CHAIDISAMBIGUATE test = CHAIDISAMBIGUATE(),
                                bool foo = Q);
#endif


#ifndef CHAI_DISABLE_RM
  /*!
   * \brief Assign a user-defined callback triggerd upon memory migration.
   *
   * The callback is a function of the form
   *
   *   void callback(chai::ExecutionSpace moved_to, size_t num_bytes)
   *
   * Where moved_to is the execution space that the data was moved to, and
   * num_bytes is the number of bytes moved.
   *
   */
  CHAI_HOST void setUserCallback(UserCallback const& cback)
  {
    if (m_pointer_record && m_pointer_record != &ArrayManager::s_null_record) {
      m_pointer_record->m_user_callback = cback;
    }
  }


private:
  /*!
   * \brief Moves the inner data of a ManagedArray.
   *
   * Called in the copy constructor of ManagedArray. In this implementation, the
   * inner type inherits from CHAICopyable, so the inner data will be moved. For
   * example, this version of the method is called when there are nested
   * ManagedArrays.
   */
  template <bool B = std::is_base_of<CHAICopyable, T>::value,
            typename std::enable_if<B, int>::type = 0>
  CHAI_HOST void moveInnerImpl() const; 

  /*!
   * \brief Does nothing since the inner data type does not inherit from
   * CHAICopyable.
   *
   * Called in the copy constructor of ManagedArray. In this implementation, the
   * inner type does not inherit from CHAICopyable, so nothing will be done. For
   * example, this version of the method is called when there are not nested
   * ManagedArrays.
   */
  template <bool B = std::is_base_of<CHAICopyable, T>::value,
            typename std::enable_if<!B, int>::type = 0>
  CHAI_HOST void moveInnerImpl() const;
#endif

public:
  CHAI_HOST_DEVICE void shallowCopy(ManagedArray<T> const& other) const
  {
    m_active_pointer = other.m_active_pointer;
    m_active_base_pointer = other.m_active_base_pointer;
    m_resource_manager = other.m_resource_manager;
    m_elems = other.m_elems;
    m_offset = other.m_offset;
    m_pointer_record = other.m_pointer_record;
    m_is_slice = other.m_is_slice;
#ifndef CHAI_DISABLE_RM
#if !defined(CHAI_DEVICE_COMPILE)
  // if we can, ensure elems is based off the pointer_record size to protect against
  // casting leading to incorrect size info in m_elems.
  if (m_pointer_record != nullptr && !m_is_slice) {
     m_elems = m_pointer_record->m_size / sizeof(T);
  }
#endif
#endif
  }

  /*!
  * Accessor for m_is_slice -whether this array was created with a slice() command.
  */
  CHAI_HOST_DEVICE bool isSlice() { return m_is_slice;}


private:
  CHAI_HOST void modify(size_t i, const T& val) const;
  // The following are only used by ManagedArray.inl, but for template
  // shenanigan reasons need to be defined here.
#if !defined(CHAI_DISABLE_RM)
  // if T is a CHAICopyable, then it is important to initialize all the
  // ManagedArrays to nullptr at allocation, since it is extremely easy to
  // trigger a moveInnerImpl, which expects inner values to be initialized.
  template <bool B = std::is_base_of<CHAICopyable, T>::value,
            typename std::enable_if<B, int>::type = 0>
  CHAI_HOST bool initInner(size_t start = 0)
  {
    for (size_t i = start; i < m_elems; ++i) {
      m_active_base_pointer[i] = nullptr;
    }
    return true;
  }

  // Do not deep initialize if T is not a CHAICopyable.
  template <bool B = std::is_base_of<CHAICopyable, T>::value,
            typename std::enable_if<!B, int>::type = 0>
  CHAI_HOST bool initInner(size_t = 0)
  {
    return false;
  }
#endif
protected:
  /*!
   * Currently active data pointer.
   */
  mutable T* m_active_pointer = nullptr;
  mutable T* m_active_base_pointer = nullptr;

  /*!
   * Pointer to ArrayManager instance.
   */
  mutable ArrayManager* m_resource_manager = nullptr;

  /*!
   * Number of elements in the ManagedArray.
   */
  mutable size_t m_elems = 0;
  mutable size_t m_offset = 0;

  /*!
   * Pointer to PointerRecord data.
   */
  mutable PointerRecord* m_pointer_record = nullptr;

  mutable bool m_is_slice = false;
};

/*!
 * \brief Construct a ManagedArray from an externally allocated pointer.
 *
 * The pointer can exist in any valid ExecutionSpace, and can either be "owned"
 * or "unowned". An owned pointer will be deallocated by the ArrayManager when
 * free is called on the returned ManagedArray object.
 *
 * \param data Pointer to the raw data.
 * \param elems Number of elements in the data pointer.
 * \param space ExecutionSpace where the data pointer exists.
 * \param owned If true, data will be deallocated by the ArrayManager.
 *
 * \tparam T Type of the raw data.
 *
 * \return A new ManagedArray containing the raw data pointer.
 */
template <typename T>
ManagedArray<T> makeManagedArray(T* data,
                                 size_t elems,
                                 ExecutionSpace space,
                                 bool owned)
{
#if !defined(CHAI_DISABLE_RM)
  ArrayManager* manager = ArrayManager::getInstance();

  // First, try and find an existing PointerRecord for the pointer
  PointerRecord* record = manager->getPointerRecord(data);
  bool existingRecord = true;
  if (record == &ArrayManager::s_null_record) {
    // create a new pointer record for external pointer
    record = manager->makeManaged(data, sizeof(T) * elems, space, owned);
    existingRecord = false;
  }
  ManagedArray<T> array(record, space);

  if (existingRecord && !owned) {
    // pointer has an owning PointerRecord, create a non-owning ManagedArray
    // slice
    array = array.slice(0, elems);
  }

  if (!std::is_const<T>::value) {
    array.registerTouch(space);
  }
#else
  PointerRecord recordTmp;
  recordTmp.m_pointers[space] = data;
  recordTmp.m_size = sizeof(T) * elems;
  recordTmp.m_owned[space] = owned;

  ManagedArray<T> array(&recordTmp, space);
#endif

  return array;
}

/*!
 * \brief Create a copy of the given ManagedArray with a single allocation in
 * the active space of the given array.
 *
 * \param array The ManagedArray to copy.
 *
 * \tparam T Type of the raw data.
 *
 * \return A copy of the given ManagedArray.
 */
template <typename T>
ManagedArray<T> deepCopy(ManagedArray<T> const& array)
{
  T* data_ptr = array.getActiveBasePointer();

  ArrayManager* manager = ArrayManager::getInstance();

  PointerRecord const* record = manager->getPointerRecord(data_ptr);

  PointerRecord* copy_record = manager->deepCopyRecord(record);

  return ManagedArray<T>(copy_record, copy_record->m_last_space);
}

template <typename T>
CHAI_INLINE CHAI_HOST_DEVICE ManagedArray<T> ManagedArray<T>::slice( size_t offset, size_t elems) const
{
  ManagedArray<T> slice;
  slice.m_resource_manager = m_resource_manager;
  if (elems == (size_t) -1) {
    elems = size() - offset;
  }
  if (offset + elems > size()) {
#if !defined(CHAI_DEVICE_COMPILE)
    CHAI_LOG(Debug,
             "Invalid slice. No active pointer or index out of bounds");
#endif
  } else {
    slice.m_pointer_record = m_pointer_record;
    slice.m_active_base_pointer = m_active_base_pointer;
    slice.m_offset = offset + m_offset;
    slice.m_active_pointer = m_active_base_pointer + slice.m_offset;
    slice.m_elems = elems;
    slice.m_is_slice = true;
  }
  return slice;
}

}  // end of namespace chai

#if defined(CHAI_DISABLE_RM)
#include "chai/ManagedArray_thin.inl"
#else
#include "chai/ManagedArray.inl"
#endif
#endif  // CHAI_ManagedArray_HPP
