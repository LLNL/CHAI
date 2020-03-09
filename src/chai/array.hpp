#ifndef CHAI_array_HPP
#define CHAI_array_HPP

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
   inline IndexWrapper(const IndexWrapper<noconst_value_type>& new_val) : val(new_val.val) {}
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
class array {
 public:
   //default constructor
   array();

   //manual destructor
   ~array() = default;
   
   //copy constructors with copy and swap idiom
   array(const array<T>& other);
   array(array<T>&& other);
   array<T>& operator=(array<T> other);
   friend inline void swap(array<T>&, array<T>&);

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

} // end of namespace chai

#include "chai/array.inl"

#endif
