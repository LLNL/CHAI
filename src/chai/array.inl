#ifndef CHAI_array_INL
#define CHAI_array_INL

#include "chai/array.hpp"

namespace chai {

// 
// array
//
template <typename TTT>
array<TTT>::array() : 
  _pimpl(nullptr),
  _activePointer(nullptr),
  _offset(0),
  _size(0)
{
}

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
} // end of namespace chai


#endif // CHAI_array_INL
