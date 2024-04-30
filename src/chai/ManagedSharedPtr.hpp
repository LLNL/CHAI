#ifndef CHAI_MANAGED_SHARED_PTR
#define CHAI_MANAGED_SHARED_PTR

#include <type_traits>

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/SharedPtrCounter.hpp"
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
  CHAI_HOST_DEVICE
  constexpr ManagedSharedPtr() noexcept : m_record_count() {}

  //// *Default* Ctor with convertible type Yp -> Tp
  template<typename Yp, typename Deleter, typename = SafeConv<Yp>> 
  ManagedSharedPtr(Yp* host_p, Yp* device_p, Deleter d) 
    : m_record_count(host_p, device_p, std::move(d))
    , m_active_pointer(m_record_count.m_get_pointer<Yp>(chai::CPU))
    , m_resource_manager(SharedPtrManager::getInstance())
  {}

  /*
   * Copy Constructors
   */
  CHAI_HOST_DEVICE
  ManagedSharedPtr(ManagedSharedPtr const& rhs) noexcept
    : m_record_count(rhs.m_record_count)
    , m_active_pointer(rhs.m_active_pointer)
    , m_resource_manager(rhs.m_resource_manager)
  {
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_active_pointer) move(ArrayManager::getInstance()->getExecutionSpace()); // TODO: Use a generic interface for RAJA queries.
    //if (m_active_pointer) move(m_resource_manager->getExecutionSpace());
#endif
  }

  template<typename Yp, typename = Compatible<Yp>>
  CHAI_HOST_DEVICE
  ManagedSharedPtr(ManagedSharedPtr<Yp> const& rhs) noexcept
    : m_record_count(rhs.m_record_count)
    , m_active_pointer(rhs.m_active_pointer)
    , m_resource_manager(rhs.m_resource_manager)
  {
#if !defined(CHAI_DEVICE_COMPILE)
    if (m_active_pointer) move(ArrayManager::getInstance()->getExecutionSpace());
    //if (m_active_pointer) move(m_resource_manager->getExecutionSpace());
#endif
  }
  
  /*
   * Accessors
   */
  CHAI_HOST_DEVICE
  element_type* get(ExecutionSpace space = chai::CPU) const noexcept { return m_active_pointer; }

  CHAI_HOST_DEVICE
  element_type& operator*() const noexcept { assert(get() != nullptr); return *get(); }

  CHAI_HOST_DEVICE
  element_type* operator->() const noexcept { assert(get() != nullptr); return get(); }

private:

  //CHAI_HOST_DEVICE
  //element_type* m_get() const noexcept { return static_cast<const ManagedSharedPtr<Tp>*>(this)->get(); }


public:
  long use_count() const noexcept { return m_record_count.m_get_use_count(); }

  CHAI_HOST
  void move(ExecutionSpace space, bool registerTouch = true) noexcept {
    printf("Calling move\n");
     ExecutionSpace prev_space = m_record_count.m_get_record()->m_last_space;
     if (prev_space == CPU || prev_space == NONE) {
        /// Move nested ManagedArrays first, so they are working with a valid m_active_pointer for the host,
        // and so the meta data associated with them are updated before we move the other array down.
        //moveInnerImpl();
     }
     m_active_pointer = static_cast<Tp*>(m_resource_manager->move((void *)m_active_pointer, m_record_count.m_get_record(), space));

     if (registerTouch) {
       m_resource_manager->registerTouch(m_record_count.m_get_record(), space);
     }
     if (space != GPU && prev_space == GPU) {
        /// Move nested ManagedArrays after the move, so they are working with a valid m_active_pointer for the host,
        // and so the meta data associated with them are updated with live GPU data
        //moveInnerImpl();
     }

  }
  /*
   * Private Members
   */
private:
  template<typename Tp1>
  friend class ManagedSharedPtr;

  msp_record_count m_record_count;
  mutable element_type* m_active_pointer = nullptr;

  mutable SharedPtrManager* m_resource_manager = nullptr;
};


template <typename T,
          typename... Args>
__global__ void msp_make_on_device(T* gpuPointer, Args... args)
{
   new(gpuPointer) T(processArguments(args)...);
}


template<typename Tp, typename... Args>
CHAI_HOST Tp* msp_make_on_device(Args... args) {
  chai::SharedPtrManager* sptr_manager = chai::SharedPtrManager::getInstance();

  auto gpu_allocator = sptr_manager->getAllocator(chai::GPU);
  Tp* gpu_ptr = static_cast<Tp*>( gpu_allocator.allocate(1*sizeof(Tp)) );

  msp_make_on_device<<<1,1>>>(gpu_ptr, args...);

  return gpu_ptr;
}

template<typename Tp, typename... Args>
ManagedSharedPtr<Tp> make_shared(Args... args) {
  Tp* gpu_pointer = make_on_device<Tp>(args...);
  Tp* cpu_pointer = make_on_host<Tp>(args...);
  std::cout << "CPU @ " << cpu_pointer << std::endl;
  std::cout << "GPU @ " << gpu_pointer << std::endl;

  return ManagedSharedPtr<Tp>(cpu_pointer, gpu_pointer, [](Tp* p){delete p;});
}

template<typename Tp, typename Deleter, typename... Args>
ManagedSharedPtr<Tp> make_shared_deleter(Args... args, Deleter d) {
  Tp* gpu_pointer = make_on_device<Tp>(args...);
  Tp* cpu_pointer = make_on_host<Tp>(args...);
  std::cout << "CPU @ " << cpu_pointer << std::endl;
  std::cout << "GPU @ " << gpu_pointer << std::endl;

  return ManagedSharedPtr<Tp>(cpu_pointer, gpu_pointer, std::move(d));
}

} // namespace chai


#endif // CHAI_MANAGED_SHARED_PTR
