#ifndef CHAI_MANAGED_SHARED_PTR
#define CHAI_MANAGED_SHARED_PTR

#include <type_traits>

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/SharedPtrCounter.hpp"
#include "chai/managed_ptr.hpp"

namespace chai {
namespace expt {


struct CHAIPoly {};

// Type traits for SFINAE
template<typename Tp, typename Yp>
struct msp_is_constructible : std::is_convertible<Yp*, Tp*>::type {};

template<typename Yp, typename Tp>
struct msp_compatible_with : std::false_type {};

template<typename Yp, typename Tp>
struct msp_compatible_with<Yp*, Tp*> : std::is_convertible<Yp*, Tp*>::type {};

template<typename Tp>
struct is_CHAICopyable : std::is_base_of<CHAICopyable, Tp>::type {};

template<typename Tp>
struct is_CHAIPoly : std::is_base_of<CHAIPoly, Tp>::type {};


template<typename Tp>
class ManagedSharedPtr : public CHAICopyable{

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
  ManagedSharedPtr(std::initializer_list<Yp*>&& ptrs,
                   std::initializer_list<ExecutionSpace>&& spaces,
                   Deleter d) 
    : m_record_count(Yp{},
        std::forward<std::initializer_list<Yp*>>(ptrs),
        std::forward<std::initializer_list<ExecutionSpace>>(spaces),
        std::move(d))
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
    if (m_active_pointer) move(ArrayManager::getInstance()->getExecutionSpace());
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
#endif
  }

  CHAI_HOST_DEVICE ManagedSharedPtr& operator=(ManagedSharedPtr const& rhs){
    m_record_count=rhs.m_record_count;
    m_active_pointer=rhs.m_active_pointer;
    m_resource_manager=rhs.m_resource_manager;

    return *this;

  }

  CHAI_HOST void swap(ManagedSharedPtr& rhs) noexcept {
    std::swap(m_active_pointer, rhs.m_active_pointer);
    std::swap(m_resource_manager, rhs.m_resource_manager);
    m_record_count.m_swap(rhs.m_record_count);
  }

  CHAI_HOST void reset() noexcept {
    ManagedSharedPtr().swap(*this);
  }

  CHAI_HOST_DEVICE void shallowCopy(ManagedSharedPtr const& rhs) {
    m_active_pointer = rhs.m_active_pointer;
    m_active_pointer=rhs.m_active_pointer;
    m_resource_manager=rhs.m_resource_manager;
  }

  
  /*
   * Accessors
   */
  CHAI_HOST_DEVICE
  const element_type* cget(ExecutionSpace space = chai::CPU) const noexcept { 
    CHAI_UNUSED_VAR(space);
#if !defined(CHAI_DEVICE_COMPILE)
  if (m_active_pointer) {
     move(space, false);
  }
#endif
    return m_active_pointer; 
  }
  CHAI_HOST_DEVICE
  element_type* get(ExecutionSpace space = chai::CPU) const noexcept { 
    CHAI_UNUSED_VAR(space);
#if !defined(CHAI_DEVICE_COMPILE)
  if (m_active_pointer) {
     move(space);
  }
#endif
    return m_active_pointer; 
  }

  CHAI_HOST_DEVICE
  element_type& operator*() const noexcept { assert(get() != nullptr); return *get(); }

  CHAI_HOST_DEVICE
  element_type* operator->() const noexcept { assert(get() != nullptr); return get(); }

public:
  long use_count() const noexcept { return m_record_count.m_get_use_count(); }

  CHAI_INLINE
  CHAI_HOST void registerTouch(ExecutionSpace space) {
    m_resource_manager->registerTouch(m_record_count.m_get_record(), space);
  }

  CHAI_HOST
  void move(ExecutionSpace space,
            bool registerTouch=(!std::is_const<Tp>::value || is_CHAICopyable<Tp>::value)) const {
     ExecutionSpace prev_space = m_record_count.m_get_record()->m_last_space;
     if (prev_space != GPU && space == GPU) {
        /// Move nested chai managed types first, so they are working with a valid m_active_pointer for the host,
        // and so the meta data associated with them are updated before we move the ManagedSharedPtr down.
       moveInnerImpl();
     }
     auto old_pointer = m_active_pointer;
     m_active_pointer = static_cast<Tp*>(m_resource_manager->move(
           (void *)m_active_pointer, m_record_count.m_get_record(), space, is_CHAIPoly<Tp>::value));
     if (old_pointer != m_active_pointer) {
     }

     if (registerTouch) {
       m_resource_manager->registerTouch(m_record_count.m_get_record(), space);
     }
     if (space != GPU && prev_space == GPU) {
       /// Move nested chai managed types after the move, so they are working with a valid m_active_pointer for the host,
       // and so the meta data associated with them are updated with live GPU data
       moveInnerImpl();
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

  template <bool B = is_CHAICopyable<Tp>::value,
            typename std::enable_if<B, int>::type = 0>
  CHAI_HOST
  void
  moveInnerImpl() const 
  {
    m_record_count.moveInnerImpl();
  }

  template <bool B = is_CHAICopyable<Tp>::value,
            typename std::enable_if<!B, int>::type = 0>
  CHAI_HOST
  void
  moveInnerImpl() const
  {
  }

};

namespace detail {

#if defined(CHAI_ENABLE_CUDA) or defined(CHAI_ENABLE_HIP)
namespace impl {

template <typename T,
          typename... Args>
__global__ void msp_make_on_device(T* gpuPointer, Args&&... args)
{
   new(gpuPointer) T(std::forward<Args>(args)...);
}

} // namespace impl

//template<typename Tp>
template<typename Tp, typename... Args>
CHAI_INLINE
CHAI_HOST Tp* msp_make_on_device(Args&&... args) {
  Tp* gpu_ptr = nullptr;
  chai::expt::SharedPtrManager* sptr_manager = chai::expt::SharedPtrManager::getInstance();

  auto gpu_allocator = sptr_manager->getAllocator(chai::GPU);
  gpu_ptr = static_cast<Tp*>( gpu_allocator.allocate(1*sizeof(Tp)) );

  impl::msp_make_on_device<<<1,1>>>(gpu_ptr, std::forward<Args>(args)...);

  return gpu_ptr;
}
#endif // defined(CHAI_ENABLE_CUDA) of defined(CHAI_ENABLE_HIP)

template<typename Tp, typename... Args>
CHAI_INLINE
CHAI_HOST Tp* msp_make_on_host(Args&&... args) {
  chai::expt::SharedPtrManager* sptr_manager = chai::expt::SharedPtrManager::getInstance();

  auto cpu_allocator = sptr_manager->getAllocator(chai::CPU);

  Tp* cpu_ptr = static_cast<Tp*>( cpu_allocator.allocate(1*sizeof(Tp)) );

  new (cpu_ptr) Tp{std::forward<Args>(args)...};

  return cpu_ptr;
}

} // namespace detail

template<typename Tp, typename... Args>
CHAI_INLINE
CHAI_HOST
ManagedSharedPtr<Tp> make_shared(Args&&... args) {
  using Tp_non_const = std::remove_const_t<Tp>;

  Tp* cpu_pointer = detail::msp_make_on_host<Tp_non_const>(std::forward<Args>(args)...);

#if defined(CHAI_ENABLE_CUDA) or defined(CHAI_ENABLE_HIP)

  Tp* gpu_pointer = detail::msp_make_on_device<Tp_non_const>();
#if defined(CHAI_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif
#if defined(CHAI_ENABLE_HIP)
  CHAI_UNUSED_VAR(hipDeviceSynchronize());
#endif

  auto result = ManagedSharedPtr<Tp>({cpu_pointer, gpu_pointer}, {CPU, GPU},
      [] CHAI_HOST_DEVICE (Tp* p){p->~Tp();}
  );

  result.registerTouch(chai::CPU);

  if (!is_CHAICopyable<Tp>::value) {
    result.move(chai::GPU, false);
    result.move(chai::CPU, false);
  }

#else // defined(CHAI_ENABLE_CUDA) or defined(CHAI_ENABLE_HIP)

  auto result = ManagedSharedPtr<Tp>({cpu_pointer}, {CPU},
      [] (Tp* p){p->~Tp();}
  );

#endif // defined(CHAI_ENABLE_CUDA) or defined(CHAI_ENABLE_HIP)

  return result;
}

} // namespace expt
} // namespace chai


#endif // CHAI_MANAGED_SHARED_PTR
