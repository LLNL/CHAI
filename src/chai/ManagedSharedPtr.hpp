#ifndef CHAI_MANAGED_SHARED_PTR
#define CHAI_MANAGED_SHARED_PTR

#include <type_traits>

#include "chai/ExecutionSpaces.hpp"
//#include "chai/SharedPtrManager.hpp"
#include "chai/SharedPointerRecord.hpp"
#include "chai/managed_ptr.hpp"

namespace chai {



class msp_counted_base {
public:
  msp_counted_base() noexcept : m_use_count(1) {}

  virtual ~msp_counted_base() noexcept {}

  virtual void m_dispose() noexcept = 0;
  virtual void m_destroy() noexcept { delete this; }

  void m_add_ref_copy() noexcept { ++m_use_count; }

  void m_release() noexcept {
    if(--m_use_count == 0) {
      m_dispose();
      m_destroy();
    }
  }

  long m_get_use_count() const noexcept { return m_use_count; }
private:
  msp_counted_base(msp_counted_base const&) = delete;
  msp_counted_base& operator=(msp_counted_base const&) = delete;

  long m_use_count = 0;
};

template<typename Ptr, typename Record>
class msp_counted_ptr final : public msp_counted_base {
public:
  msp_counted_ptr(Record p) noexcept : m_record(p) {}
  //virtual void m_dispose() noexcept { delete (m_record.get_pointer(chai::CPU)); }// TODO : Other Exec spaces...
  virtual void m_dispose() noexcept { delete (Ptr)m_record->m_pointers[chai::CPU]; }// TODO : Other Exec spaces...
  virtual void m_destroy() noexcept { delete this; }
  msp_counted_ptr(msp_counted_ptr const&) = delete;
  msp_counted_ptr& operator=(msp_counted_ptr const&) = delete;
private:
  Record m_record;
};

template<typename Ptr, typename Record, typename Deleter>
class msp_counted_deleter final : public msp_counted_base {

  class impl {
  public:
    impl(Record p, Deleter d) : m_record(p), m_deleter(std::move(d)) {}
    Deleter& m_del() noexcept { return m_deleter; }
    Record m_record;
    Deleter m_deleter;
  };

public:
  msp_counted_deleter(Record p, Deleter d) noexcept : m_impl(p, std::move(d)) {}
  virtual void m_dispose() noexcept { 
    printf("Delete GPU Memory Here...\n");
    m_impl.m_del()((Ptr)m_impl.m_record->m_pointers[chai::CPU]);
  }
  virtual void m_destroy() noexcept { this->~msp_counted_deleter(); }
  msp_counted_deleter(msp_counted_deleter const&) = delete;
  msp_counted_deleter& operator=(msp_counted_deleter const&) = delete;
private:
  impl m_impl;
};


class msp_shared_count {
public:
  constexpr msp_shared_count() noexcept : m_pi(0) {}

  template<typename Ptr, typename Record>
  explicit msp_shared_count(Ptr, Record p)
  : m_pi( new  msp_counted_ptr<Ptr, Record>(p) ) {}

  template<typename Ptr, typename Record, typename Deleter>
  explicit msp_shared_count(Ptr, Record p, Deleter d)
  : m_pi( new  msp_counted_deleter<Ptr, Record, Deleter>(p, d) ) {}

  ~msp_shared_count() noexcept
  { if (m_pi) m_pi->m_release(); }

  msp_shared_count(msp_shared_count const& rhs) noexcept : m_pi(rhs.m_pi)
  { if (m_pi) m_pi->m_add_ref_copy(); }

  msp_shared_count& operator=(msp_shared_count const& rhs) noexcept {
    msp_counted_base* temp = rhs.m_pi;
    if (temp != m_pi)
    {
      if (temp) temp->m_add_ref_copy();
      if (m_pi) m_pi->m_release();
      m_pi = temp;
    }
    return *this;
  }

  void m_swap(msp_shared_count& rhs) noexcept {
    msp_counted_base* temp = rhs.m_pi;
    rhs.m_pi = m_pi;
    m_pi = temp;
  }

  long m_get_use_count() const noexcept 
  { return m_pi ? m_pi->m_get_use_count() : 0; }

  friend inline bool
  operator==(msp_shared_count const& a, msp_shared_count const& b) noexcept
  { return a.m_pi == b.m_pi; }

  msp_counted_base* m_pi;

};






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
  constexpr ManagedSharedPtr() noexcept : m_ref_count() {}

  //// *Default* Ctor with convertible type Yp -> Tp
  //template<typename Yp, typename = SafeConv<Yp>>
  //explicit ManagedSharedPtr(Yp* host_p) :
  //  m_pointer_record(new msp_pointer_record<Tp>(host_p)),
  //  m_ref_count(host_p, m_pointer_record),
  //  m_active_pointer(static_cast<Yp*>(m_pointer_record->m_pointers[chai::CPU]))
  //  //m_resource_manager(SharedPtrManager::getInstance())
  //{}

  //template<typename Yp, typename = SafeConv<Yp>>
  //explicit ManagedSharedPtr(Yp* host_p, Yp* device_p) :
  //  m_pointer_record(new msp_pointer_record<Yp>(host_p, device_p)),
  //  m_ref_count(host_p, m_pointer_record),
  //  m_active_pointer(static_cast<Yp*>(m_pointer_record->m_pointers[chai::CPU]))
  //  //m_resource_manager(SharedPtrManager::getInstance())
  //{}

  //template<typename Yp, typename Deleter, typename = SafeConv<Yp>> 
  //ManagedSharedPtr(Yp* host_p, Deleter d) :
  //  m_pointer_record(new msp_pointer_record<Yp>(host_p)),
  //  m_ref_count(host_p, m_pointer_record, std::move(d)),
  //  m_active_pointer(static_cast<Yp*>(m_pointer_record->m_pointers[chai::CPU]))
  //  //m_resource_manager(SharedPtrManager::getInstance())
  //{}

  template<typename Yp, typename Deleter, typename = SafeConv<Yp>> 
  ManagedSharedPtr(Yp* host_p, Yp* device_p, Deleter d) :
    m_pointer_record(new msp_pointer_record<Yp>(host_p, device_p)),
    m_ref_count(host_p, m_pointer_record, std::move(d)),
    m_active_pointer(static_cast<Yp*>(m_pointer_record->m_pointers[chai::CPU]))
    //m_resource_manager(SharedPtrManager::getInstance())
  {}

  /*
   * Copy Constructors
   */
  ManagedSharedPtr(ManagedSharedPtr const&) noexcept = default; // TODO: this is *NOT* going to be default

  template<typename Yp, typename = Compatible<Yp>>
  ManagedSharedPtr(ManagedSharedPtr<Yp> const& rhs) noexcept : 
    m_ref_count(rhs.m_ref_count),
    m_active_pointer(rhs.m_active_pointer)
  {
    // TODO : Is this safe??
    m_pointer_record = reinterpret_cast<msp_pointer_record<Tp>*>(rhs.m_pointer_record);
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
  long use_count() const noexcept { return m_ref_count.m_get_use_count(); }

  /*
   * Private Members
   */
private:
  template<typename Tp1>
  friend class ManagedSharedPtr;

  //template<typename Yp, typename... Args>
  //friend ManagedSharedPtr<Yp> make_managed(Args... args);
  
  mutable msp_pointer_record<Tp>* m_pointer_record = nullptr;
  msp_shared_count m_ref_count;
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
