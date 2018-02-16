/*
 * ManagedMultiArray.h
 *
 *  Created on: Feb 7, 2018
 *      Author: settgast1
 */

#ifndef SRC_UTIL_MANAGEDMULTIARRAY_H_
#define SRC_UTIL_MANAGEDMULTIARRAY_H_
#include "ManagedArray.hpp"

namespace chai
{

template< typename T, int NDIMS >
class ManagedMultiArray
{
public:
  using value_type = T;
  using size_t = std::size_t;
  using pointer = T*;
  using const_pointer = T const *;
  using reference = T&;
  using const_reference = T const &;


  CHAI_HOST_DEVICE ManagedMultiArray();

  template< typename... DIMS >
  CHAI_HOST_DEVICE ManagedMultiArray( ExecutionSpace space, DIMS...dims );

  template< typename... DIMS >
  CHAI_HOST_DEVICE ManagedMultiArray( DIMS...dims );


  CHAI_HOST_DEVICE ManagedMultiArray( ManagedMultiArray const & source );


  ManagedMultiArray & operator=( ManagedMultiArray const & source ) = delete;
  ManagedMultiArray( ManagedMultiArray && ) = delete;
  ManagedMultiArray & operator=( ManagedMultiArray && ) = delete;


  template< typename... DIMS>
  CHAI_HOST void allocate( ExecutionSpace space, DIMS... dims );

  template< typename... DIMS>
  CHAI_HOST void allocate( DIMS... dims );

  template< typename... DIMS>
  CHAI_HOST void reallocate(DIMS... dims);

  CHAI_HOST void free();

  CHAI_HOST void reset();

  template< typename... DIMS>
  void resize( DIMS... newDims );

//  template< typename... DIMS>
//  void reserve( DIMS... newDims );

  CHAI_HOST size_t size() const;

  CHAI_HOST size_t size( int dim ) const;

  template<typename Idx>
  CHAI_HOST_DEVICE T& operator[](const Idx i) const;


private:

  template< typename...DIMS >
  size_t calculateNewSize(DIMS... dims, size_t * const tempDims );

  size_t m_dims[NDIMS];

  ManagedArray<T> m_managedArray;

  template< int INDEX, typename DIM0, typename... DIMS >
  struct dimension_unpacker;

  template< typename DIM0, typename... DIMS >
  struct dimension_unpacker<1,DIM0,DIMS...> ;
//  template< int DIM >
//  struct size_helper;
};

} /* namespace chai */

#include "ManagedMultiArray_inl.hpp"

#endif /* SRC_UTIL_MANAGEDMULTIARRAY_H_ */
