/*
 * ManagedMultiArray.cpp
 *
 *  Created on: Feb 7, 2018
 *      Author: settgast1
 */

#include "ManagedMultiArray.hpp"

namespace chai
{

template< typename T, int NDIMS >
template< int INDEX, typename DIM0, typename... DIMS >
struct ManagedMultiArray<T,NDIMS>::dimension_unpacker
{
  CHAI_INLINE
  CHAI_HOST_DEVICE
  static void f( size_t * m_dims, DIM0 dim0, DIMS... dims )
  {
    m_dims[INDEX-1] = dim0;
    dimension_unpacker< INDEX-1, DIMS...>::f( m_dims, dims... );
  }
};

template< typename T, int NDIMS >
template< typename DIM0, typename... DIMS >
struct ManagedMultiArray<T,NDIMS>::dimension_unpacker<1,DIM0,DIMS...>
{
  CHAI_INLINE
  CHAI_HOST_DEVICE
  static void f( size_t * m_dims, DIM0 dim0, DIMS... )
  {
    m_dims[0] = dim0;
  }
};

//template< typename T, int NDIMS >
//template< int DIM >
//struct ManagedMultiArray<T,NDIMS>::size_helper
//{
//  template< int INDEX=DIM >
//  constexpr static typename std::enable_if<INDEX!=NDIMS-1,size_t>::type
//  f( INDEX_TYPE const * const restrict dims )
//  {
//    return dims[INDEX] * size_helper<INDEX+1>::f(dims);
//  }
//
//  template< int INDEX=DIM >
//  constexpr static typename std::enable_if<INDEX==NDIMS-1,size_t>::type
//  f( INDEX_TYPE const * const restrict dims )
//  {
//    return dims[INDEX];
//  }
//};

template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedMultiArray<T,NDIMS>::ManagedMultiArray():
  m_dims(),
  m_managedArray()
{}

template< typename T, int NDIMS >
template< typename... DIMS >
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedMultiArray<T,NDIMS>::ManagedMultiArray( ExecutionSpace space, DIMS...dims ):
  m_dims{static_cast<size_t>(dims)...},
  m_managedArray()
{
  allocate<DIMS...>( space, dims... );
}

template< typename T, int NDIMS >
template< typename... DIMS >
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedMultiArray<T,NDIMS>::ManagedMultiArray( DIMS...dims ):
  m_dims(),
  m_managedArray()
{
  allocate<DIMS...>( chai::NONE, dims... );
}

template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST_DEVICE
ManagedMultiArray<T,NDIMS>::ManagedMultiArray( ManagedMultiArray const & source ):
  m_dims(),
  m_managedArray(source.m_managedArray)
{
  for( int a=0 ; a<NDIMS ; ++a )
  {
    m_dims[a]     = source.m_dims[a];
  }
}


template< typename T, int NDIMS >
template< typename... DIMS>
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::allocate( ExecutionSpace space, DIMS... newDims )
{
  static_assert( sizeof ... (DIMS) == NDIMS,
                 "Error: calling ManagedMultiArray::allocate( ExecutionSpace space, DIMS... newDims ) "
                 "with incorrect number of arguments.");

  size_t newSize = calculateNewSize<DIMS...>(newDims..., m_dims);

  m_managedArray.allocate( newSize, space );
}

template< typename T, int NDIMS >
template< typename... DIMS>
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::allocate( DIMS... newDims )
{
  allocate<DIMS...>( chai::CPU, newDims...);
}

template< typename T, int NDIMS >
template< typename... DIMS>
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::reallocate( DIMS... newDims )
{
  static_assert( sizeof ... (DIMS) == NDIMS,
                 "Error: calling ManagedMultiArray::allocate( ExecutionSpace space, DIMS... newDims ) "
                 "with incorrect number of arguments.");

  size_t newSize = calculateNewSize<DIMS...>(newDims..., m_dims);

  m_managedArray.reallocate( newSize );
}


template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::free()
{
  m_managedArray.free();
}

template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::reset()
{
  m_managedArray.reset();
}


template< typename T, int NDIMS >
template< typename... DIMS>
CHAI_INLINE
CHAI_HOST
void ManagedMultiArray<T,NDIMS>::resize( DIMS... newDims )
{
  if( size()==0 )
  {
    allocate(newDims...);
  }
  else
  {
    reallocate(newDims...);
  }

}

//template< typename T, int NDIMS >
//template< typename... DIMS>
//CHAI_INLINE
//CHAI_HOST_DEVICE
//void ManagedMultiArray<T,NDIMS>::reserve( DIMS... newDims )
//{
//}



template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST
size_t ManagedMultiArray<T,NDIMS>::size() const
{
  size_t rval = 1;
  for( int dim=0 ; dim<NDIMS ; ++dim )
  {
    rval *= m_dims[dim];
  }
  return rval;
}

template< typename T, int NDIMS >
CHAI_INLINE
CHAI_HOST
size_t ManagedMultiArray<T,NDIMS>::size( int const dim ) const
{
  return m_dims[dim];
}

template<typename T, int NDIMS>
template<typename INDEX>
CHAI_INLINE
CHAI_HOST_DEVICE T& ManagedMultiArray<T,NDIMS>::operator[](const INDEX i) const {
  return m_managedArray[i];
}







template< typename T, int NDIMS >
template< typename... DIMS>
CHAI_INLINE
CHAI_HOST
size_t ManagedMultiArray<T,NDIMS>::calculateNewSize( DIMS... newDims, size_t * const tempDims )
{
  dimension_unpacker<NDIMS,DIMS...>::f( m_dims, newDims...);

  size_t newSize = 1;
  for( int dim=0 ; dim<NDIMS ; ++dim )
  {
    newSize *= tempDims[dim];
  }

  return newSize;
}



} /* namespace chai */
