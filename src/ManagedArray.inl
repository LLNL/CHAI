#ifndef CHAI_ManagedArray_INL
#define CHAI_ManagedArray_INL

#include "ManagedArray.hpp"
#include "ArrayManager.hpp"

namespace chai {

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray():
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(0)
{
  m_resource_manager = ArrayManager::getInstance();
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(
    uint elems, ExecutionSpace space):
  m_active_pointer(nullptr),
  m_resource_manager(nullptr),
  m_elems(elems)
{
  m_resource_manager = ArrayManager::getInstance();
  this->allocate(elems, space);
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(ManagedArray const& other):
  m_active_pointer(other.m_active_pointer),
  m_resource_manager(other.m_resource_manager),
  m_elems(other.m_elems)
{
#if !defined(__CUDA_ARCH__)
  CHAI_LOG("ManagedArray", "Moving " << m_active_pointer);

  m_active_pointer = static_cast<T*>(m_resource_manager->move(const_cast<T_non_const*>(m_active_pointer)));

  CHAI_LOG("ManagedArray", "Moved to " << m_active_pointer);

  /*
   * Register touch
   */
  if (!std::is_const<T>::value) {
    CHAI_LOG("ManagedArray", "T is non-const, registering touch of pointer" << m_active_pointer);
    T_non_const* non_const_pointer = const_cast<T_non_const*>(m_active_pointer);
    m_resource_manager->registerTouch(non_const_pointer);
  }
#endif
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::ManagedArray(T* data, ArrayManager* array_manager, uint elems) :
  m_active_pointer(data), 
  m_resource_manager(array_manager),
  m_elems(elems)
{
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::allocate(uint elems, ExecutionSpace space) {
  CHAI_LOG("ManagedArray", "Allocating array of size " << elems << " in space " << space);

  m_elems = elems;
  m_active_pointer = static_cast<T*>(m_resource_manager->allocate<T>(elems, space));

  CHAI_LOG("ManagedArray", "m_active_ptr allocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::reallocate(uint elems)
{
  CHAI_LOG("ManagedArray", "Reallocating array of size " << m_elems << " with new size" << elems);

  m_elems = elems;
  m_active_pointer = static_cast<T*>(m_resource_manager->reallocate<T>(m_active_pointer, elems));

  CHAI_LOG("ManagedArray", "m_active_ptr reallocated at address: " << m_active_pointer);
}

template<typename T>
CHAI_INLINE
CHAI_HOST void ManagedArray<T>::free()
{
  m_resource_manager->free(m_active_pointer);
}


template<typename T>
CHAI_INLINE
CHAI_HOST uint ManagedArray<T>::size() const {
  return m_elems;
}

template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedArray<T>::operator[](const int i) const {
  return m_active_pointer[i];
}

// template<typename T>
// CHAI_INLINE
// CHAI_HOST_DEVICE T& ManagedArray<T>::pick(size_t i, T_non_const& val) {
// #ifdef __CUDA_ARCH__
//           val = m_active_ptr[i]; 
// #else
// #endif
// 
// }



template<typename T>
CHAI_INLINE
CHAI_HOST_DEVICE ManagedArray<T>::operator T*() const {
#if !defined(__CUDA_ARCH__)
  m_resource_manager->setExecutionSpace(CPU);

  m_active_pointer = static_cast<T*>(m_resource_manager->move(m_active_pointer));

  m_resource_manager->registerTouch(m_active_pointer);

  return m_active_pointer;
#else
  return m_active_pointer;
#endif
}

template<typename T>
template<bool B,typename std::enable_if<!B, int>::type>
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedArray<T>::operator ManagedArray<const T> () const
{
  return ManagedArray<const T>(const_cast<const T*>(m_active_pointer), m_resource_manager, m_elems);
}

} // end of namespace chai

#endif // CHAI_ManagedArray_INL
