#ifndef CHAI_MANAGED_SHARED_PTR
#define CHAI_MANAGED_SHARED_PTR

#include <type_traits>

#include "chai/ExecutionSpaces.hpp"
//#include "chai/SharedPtrManager.hpp"
#include "chai/SharedPointerRecord.hpp"
#include "chai/managed_ptr.hpp"

namespace chai {







// Type traits for SFINAE
template<typename Tp, typename Yp>
struct msp_is_constructible : std::is_convertible<Yp*, Tp*>::type {};

template<typename Yp, typename Tp>
struct msp_compatible_with : std::false_type {};

template<typename Yp, typename Tp>
struct msp_compatible_with<Yp*, Tp*> : std::is_convertible<Yp*, Tp*>::type {};




template<typename Tp>
class ManagedSharedPtr {

public:
  using element_type = Tp;//typename std::remove_extent<Tp>::type;

private:
  template<typename Yp>
  using SafeConv = typename std::enable_if<
                              msp_is_constructible<Tp, Yp>::value
                            >::type;

  template<typename Yp, typename Res = void>
  using Compatible = typename std::enable_if<
                                msp_compatible_with<Yp*, Tp*>::value,
                                Res
                              >::type;

  template<typename Yp>
  using Assignable = Compatible<Yp, ManagedSharedPtr&>;

public:

  /*
   * Constructors
   */
  constexpr ManagedSharedPtr() noexcept : m_record_count() {}

  //// *Default* Ctor with convertible type Yp -> Tp
  template<typename Yp, typename Deleter, typename = SafeConv<Yp>> 
  ManagedSharedPtr(Yp* host_p, Yp* device_p, Deleter d) :
    m_record_count(host_p, device_p, std::move(d)),
    m_active_pointer(m_record_count.getPointer<Yp>(chai::CPU))
  {}

  /*
   * Copy Constructors
   */
  ManagedSharedPtr(ManagedSharedPtr const&) noexcept = default; // TODO: this is *NOT* going to be default

  template<typename Yp, typename = Compatible<Yp>>
  ManagedSharedPtr(ManagedSharedPtr<Yp> const& rhs) noexcept : 
    m_record_count(rhs.m_record_count),
    m_active_pointer(rhs.m_active_pointer)
  {
  }

  
  /*
   * Accessors
   */
  element_type* get(ExecutionSpace space = chai::CPU) const noexcept { return m_active_pointer; }

  element_type& operator*() const noexcept { assert(m_get() != nullptr); return *m_get(); }

  element_type* operator->() const noexcept { assert(m_get() != nullptr); return m_get(); }

private:
  element_type* m_get() const noexcept { return static_cast<const ManagedSharedPtr<Tp>*>(this)->get(); }


public:
  long use_count() const noexcept { return m_record_count.m_get_use_count(); }

  /*
   * Private Members
   */
private:
  template<typename Tp1>
  friend class ManagedSharedPtr;

  msp_record_count m_record_count;
  mutable element_type* m_active_pointer = nullptr;
  //mutable SharedPtrManager* m_resource_manager = nullptr;
};

template<typename Tp, typename... Args>
ManagedSharedPtr<Tp> make_shared(Args... args) {
  Tp* gpu_pointer = make_on_device<Tp>(args...);
  Tp* cpu_pointer = make_on_host<Tp>(args...);

  return ManagedSharedPtr<Tp>(cpu_pointer, gpu_pointer, [](Tp* p){delete p;});
}

template<typename Tp, typename Deleter, typename... Args>
ManagedSharedPtr<Tp> make_shared_deleter(Args... args, Deleter d) {
  Tp* gpu_pointer = make_on_device<Tp>(args...);
  Tp* cpu_pointer = make_on_host<Tp>(args...);

  return ManagedSharedPtr<Tp>(cpu_pointer, gpu_pointer, std::move(d));
}

} // namespace chai


#endif // CHAI_MANAGED_SHARED_PTR
