//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_managed_array_HPP
#define CHAI_managed_array_HPP

namespace chai {

#if DEBUG
namespace private
{
   template< class T > struct remove_const          { typedef T type; };
   template< class T > struct remove_const<const T> { typedef T type; };
}
//This is a magic class that is like an int and casts down to an int,
//but int doesn't cast to it.  It is viral, and any op with it becomes it.
template <typename Base>
class IndexWrapper {
 public:
   typedef Base value_type;
   typedef typename private::remove_const<Base>::type noconst_value_type;
   explicit inline IndexWrapper(Base new_val) : val(new_val) {}
   inline IndexWrapper(const IndexWrapper<noconst_value_type>& new_val) : val(new_val.val) {}jj
   inline IndexWrapper<Base>& operator++() { ++val; return *this; }
   inline IndexWrapper<Base>& operator--() { --val; return *this; }
   inline IndexWrapper<Base> operator++(int) { return IndexWrapper<Base>(val++); }
   inline IndexWrapper<Base> operator--(int) { return IndexWrapper<Base>(val--); }
   inline operator Base() const { return val; }
   inline operator Base() { return val; }
   template <typename UUU, typename TTT=Base>
   typename std::enable_if<!std::is_const<TTT>::value,IndexWrapper<TTT>&>::type
   operator=(const UUU new_val) { val = new_val; return *this; }
   Base val;
};
   
#define _UNARY_OP(op)                                               \
template <typename Base>                                                \
inline IndexWrapper<Base> operator op(const IndexWrapper<Base> a) { return IndexWrapper<Base>(op a.val); }
_UNARY_OP(-)
_UNARY_OP(!)
#undef _UNARY_OP

#define _BINARY_OP(op)                                              \
template <typename BaseA, typename BaseB>                               \
inline auto operator op(const IndexWrapper<BaseA> a, const IndexWrapper<BaseB> b) -> IndexWrapper<decltype(a.val op b.val)> { return IndexWrapper<decltype(a.val op b.val)>(a.val op b.val); } \
template <typename BaseA, typename BaseB>                               \
inline auto operator op(const BaseA a, const IndexWrapper<BaseB> b) -> IndexWrapper<decltype(a op b.val)> { return IndexWrapper<decltype(a op b.val)>(a op b.val); } \
template <typename BaseA, typename BaseB>                               \
inline auto operator op(const IndexWrapper<BaseA> a, const BaseB b) -> IndexWrapper<decltype(a.val op b)> { return IndexWrapper<decltype(a.val op b)>(a.val op b); }
_BINARY_OP(+)
_BINARY_OP(-)
_BINARY_OP(*)
_BINARY_OP(/)
_BINARY_OP(%)
_BINARY_OP(<)
_BINARY_OP(>)
_BINARY_OP(<=)
_BINARY_OP(>=)
_BINARY_OP(==)
_BINARY_OP(!=)
_BINARY_OP(&)
_BINARY_OP(|)
_BINARY_OP(^)
#undef _BINARY_OP

#define _ASSIGN_OP(op)                                              \
template <typename Base, typename Other>                                \
inline IndexWrapper<Base>& operator op##=(IndexWrapper<Base>& a, const IndexWrapper<Other> b) { return (a = a op b); } \
template <typename Base, typename Other>                                \
inline IndexWrapper<Base>& operator op##=(IndexWrapper<Base>& a, const Other b) { return (a = a op b); }
_ASSIGN_OP(+)
_ASSIGN_OP(-)
_ASSIGN_OP(*)
_ASSIGN_OP(%)
_ASSIGN_OP(/)
_ASSIGN_OP(&)
_ASSIGN_OP(|)
_ASSIGN_OP(^)
#undef _ASSIGN_OP


//This is the type you should use in your RAJA lambdas to have type-safe chai lookups.
typedef IndexWrapper<int> index;

//This is a type that disables reading from write-only arrays.
template <typename TTT>
class OnlyAssignable
{
 public:
   inline OnlyAssignable(TTT& dataPoint) : dataPoint_(dataPoint) {}
   inline TTT& operator=(const TTT value)
   {
      dataPoint_ = value;
      return dataPoint_;
   }
 private:
   TTT& dataPoint_;
};
#else

template <typename Base>
using IndexWrapper<Base> = Base;

typedef int index;

template <typename TTT>
using OnlyAssignable<TTT> = TTT;

#endif



template <typename T>
class array_impl {
 public:
   array_impl(context_manager* cm = g_context_manager);
   ~array_impl();
   array_impl(const array_impl<T>& other);
   array_impl(array_impl<T>&& other);
   array_impl<T>& operator=(array_impl<T> other);
   friend void swap(array_impl<T>&, array_impl<T>&);

   array_impl(int size);
   
   ExecutionSpace getContext();
   bool hasContext();
   TTT* readwrite(ExecutionSpace, std::ptrdiff_t offset);
   const TTT* readonly(ExecutionSpace, std::ptrdiff_t offset);
   TTT* writeonly(ExecutionSpace, std::ptrdiff_t offset);
   TTT* overwriteall(ExecutionSpace, std::ptrdiff_t offset);

 private:
   T* _pointerRecord[NUMSPACES];
   bool _isValid[NUMSPACES];
   std::size_t _size;
};

/* This class will be the replacement for managed arrays, and it
   will have very strange copy semantics:
   - Copy constructor invoked on CPU, outside of RAJA lambda capture:
        ==> work like a shared pointer
   - Copy constructor invoked on CPU, part of RAJA lambda capture:
        ==> allocate space for the memory on the execution space if needed.
        ==> copy the data across the device boundary if needed, according to the semantics of the array.
        ==> Adjust pointers so object can be used on ExecutionSpace
   - Copy constructor invoked on GPU
        ==> Just do a straight up copy, don't increment shared ptrs
   - Destructor invoked on CPU
        ==> Decrement shared pointers, remove if necessary.
   - Destructor invoked on GPU
        ==> Do nothing.
*/
template<typename T>
class managed_array {
 public:
   //default constructor
   managed_array();

   //manual destructor
   ~managed_array() = default;
   
   //copy constructors with copy and swap idiom
   managed_array(const managed_array<T>& other);
   managed_array(managed_array<T>&& other);
   managed_array<T>& operator=(managed_array<T> other);
   friend inline void swap(managed_array<T>&, managed_array<T>&);

   //constructor with explicit size
   array(const int newSize);
   
   //Used for manual data migration
   TTT* use_on(const ExecutionSpace);
   const TTT* use_on(const ExecutionSpace) const;
   //Used for slicing
   array<TTT> slice(const std::size_t begin, const std::size_t end);
   ro_array<TTT> slice(const std::size_t begin, const std::size_t end) const;

   //used for delayed initialization.
   void resize();
   void clear();
   
   //does lookups
   template <typename Index>
   inline IndexWrapper<TTT&> operator[](const IndexWrapper<Index>);
   template <typename Index>
   inline IndexWrapper<const TTT&> operator[](const IndexWrapper<Index>) const;
   
   std::size_t size() const;
   bool empty() const;
   
   //conversion operators
   ro_array<T> readonly() const;
   wo_array<T> writeonly();
   oa_array<T> overwiteall();

   inline operator ro_array<T>() const;
   inline operator wo_array<T>();
   
   const T* readonly(ExecutionSpace) const;
   T* writeonly(ExecutionSpace);
   T* overwiteall(ExecutionSpace);

 private:
   array(std::shared_ptr<array_impl<T>> newPimpl, T* newActivePointer, std::ptrdiff_t newOffset, std::size_t newSize);
   mutable std::shared_ptr<array_impl<T>> _pimpl;
   mutable T* _activePointer;
   std::ptrdiff_t _offset;
   std::size_t _size;
};

template <typename TTT>
array<TTT>::array() : _pimpl(nullptr), _activePointer(nullptr), _offset(0), _size(0) {}

template <typename TTT>
array<TTT>& operator=(array<TTT> other) {swap(*this,other); return *this;}

template <typename TTT>
array<TTT>::array(array<TTT>&& other) {swap(*this,other);}

template <typename TTT>
inline void swap(array<TTT>& aaa, array<TTT>& bbb) {
   using std::swap;
   swap(aaa._pimpl, bbb._pimpl);
   swap(aaa._activePointer, bbb._activePointer);
   swap(aaa._offset, bbb._offset);
   swap(aaa._size, bbb._size);
}

template <typename TTT>
array<TTT>::array(const array<TTT>& other) {
   _activePointer = other._activePointer;
   _offset = other._offset;
   _size = other._size;
#if defined(__CUDA_ARCH__)
   _pimpl = nullptr;
#else
   _pimpl = other._pimpl;
   if (_pimpl && _pimpl->hasContext()) {
      //clear out pimpl if the new space doesn't share vitual memory
      //with us, or if we don't want the shared pointers updated.
      //Typically we would use this when shipping something to a RAJA
      //call, or to CUDA.
      use_on(_pimpl->getContext());
      _pimpl = nullptr;
   }
#endif
}

template <typename TTT>
array<TTT>::array(std::shared_ptr<array_impl<TTT>> newPimpl, TTT* newActivePointer, std::ptrdiff_t newOffset, std::size_t newSize) {
   _pimpl = newPimpl;
   _activePointer = newActivePointer;
   _offset = newOffset;
   _size = newSize;
}

TTT* array<TTT>::use_on(const ExecutionSpace space) {
   if (_pimpl) {
      _activePointer == _pimpl->readwrite(space, _offset);
   }
   return _activePointer;
}
const TTT* array<TTT>::use_on(const ExecutionSpace space) const {
   if (_pimpl) {
      _activePointer == _pimpl->readonly(space, _offset);
   }
   return _activePointer;
}

//Used for slicing
array<TTT> array<TTT>::slice(const std::size_t begin, const std::size_t end) {
   return array<TTT>(_pimpl,_activePointer ? _activePointer+begin : _activePointer, (nullPointer+begin) - nullPointer, end-begin);
}
ro_array<TTT> array<TTT>::slice(const std::size_t begin, const std::size_t end) const {
   return ro_array<TTT>(_pimpl,_activePointer ? _activePointer+begin : _activePointer, (nullPointer+begin) - nullPointer, end-begin);
}
   
template <typename TTT, typename Index>
inline IndexWrapper<TTT&> array<TTT>::operator[](const IndexWrapper<Index> iii) {
   return _activePointer[Index(iii)];
}
template <typename TTT, typename Index>
inline IndexWrapper<const TTT&> array<TTT>::operator[](const IndexWrapper<Index> iii) const {
   return _activePointer[Index(iii)];
}

template <typename TTT>
inline std::size_t array<TTT>::size() const { return _size; }
template <typename TTT>
inline bool array<TTT>::empty() const { return _size==0; }

template <typename TTT>
inline void array<TTT>::clear() {
   _pimpl = nullptr;
   _activePointer = nullptr;
   _offset = 0;
   _size = 0;
}

template <typename TTT>
inline void array<TTT>::array(const int newSize) : array() { resize(newSize); }

template <typename TTT>
inline void array<TTT>::resize(const int newSize) {
   _pimpl = make_shared<array_impl<TTT>>(newSize);
   _size = newSize;
   _offset = 0;
   _activePointer = nullptr;
}

template <typename TTT>
using rw_array<TTT> = array<TTT>;


/* These are just different use cases for the arrays above. They refer
   to the same data as a normal array, but have slightly different
   copy semantics within a RAJA capture or when used:

   - ro: copy data to this device if needed
   - rw: copy data to this device if needed, mark all other devices as dirty
   - wo: copy data to this device if needed, mark all other devices as dirty
   - oa:                                     mark all other devices as dirty

*/ 
template<typename TTT>
class ro_array {
 public:
   //default constructor
   ro_array();

   //manual destructor
   ~ro_array();

   //copy constructors with copy and swap idiom
   ro_array(const ro_array<TTT>& other);
   ro_array(ro_array<TTT>&& other);
   ro_array<TTT>& operator=(ro_array<TTT> other);
   friend inline void swap(ro_array<TTT>&, ro_array<TTT>&);   
   
   const TTT* use_on(const ExecutionSpace) const;
   ro_array<TTT> slice(const std::size_t begin, const std::size_t end) const;

   template <typename Index>
   inline IndexWrapper<const TTT&> operator[](const IndexWrapper<Index>) const;
   
   std::size_t size() const;
};

template<typename TTT>
class wo_array {
 public:
   //default constructor
   wo_array();

   //manual destructor
   ~wo_array();
   
   //copy constructors with copy and swap idiom
   wo_array(const wo_array<TTT>& other);
   wo_array(wo_array<TTT>&& other);
   wo_array<TTT>& operator=(wo_array<TTT> other);
   friend inline void swap(wo_array<TTT>&, wo_array<TTT>&);

   TTT* use_on(const ExecutionSpace);
   wo_array<TTT> slice(const std::size_t begin, const std::size_t end);

   template <typename Index>
   inline OnlyAssignable<TTT&> operator[](const IndexWrapper<Index>);
   
   std::size_t size() const;

   oa_array<TTT> overwiteall();
   TTT* overwiteall(ExecutionSpace);
   inline operator oa_array<TTT>();
};

template<typename TTT>
class oa_array {
 public:
   //default constructor
   oa_array();

   //manual destructor
   ~oa_array();

   //copy constructors with copy and swap idiom
   oa_array(const oa_array<TTT>& other);
   oa_array(oa_array<TTT>&& other);
   oa_array<TTT>& operator=(oa_array<TTT> other);
   friend inline void swap(oa_array<TTT>&, oa_array<TTT>&);

   TTT* use_on(const ExecutionSpace);
   oa_array<TTT> slice(const std::size_t begin, const std::size_t end);

   template <typename Index>
   inline OnlyAssignable<TTT&> operator[](const IndexWrapper<Index>);
   
   std::size_t size() const;
};
   

} // end of namespace chai

#endif