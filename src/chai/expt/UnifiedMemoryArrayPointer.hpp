#ifndef CHAI_UNIFIED_MEMORY_POINTER_HPP
#define CHAI_UNIFIED_MEMORY_POINTER_HPP

#include "chai/expt/UnifiedMemoryManager.hpp"

#include <cstddef>
#include <stdexcept>

namespace chai {
namespace expt {
  /*!
   * \class UnifiedMemoryPointer
   *
   * \brief A data structure for managing the lifetime and coherence of a
   *        unified memory array, meaning an array with a single address
   *        that is accessible from all processors/devices in a system.
   *
   * This data structure is designed for use with the ExecutionContextManager.
   * When the execution context is set and the copy constructor is triggered,
   * the underlying data is made coherent in the current execution context.
   *
   * This lends itself particularly well to an elegant programming model
   * where loop bodies are replaced with lambda expressions that capture
   * variables in the current scope by copy. If the execution context has
   * already been set, the lambda expression can be executed in that context.
   * Otherwise, the execution context must be set and then a copy of the
   * lambda expression (which triggers a copy of all the captured variables)
   * can be executed in that context. This latter case is how CHAI works
   * with RAJA via a RAJA plugin.
   *
   * This model works well for APUs where the CPU and GPU have the same
   * physical memory. It also works for pinned (i.e. page-locked) memory.
   *
   * Example using RAJA and CHAI:
   *
   * \code
   * #include "chai/UnifiedMemoryPointer.hpp"
   * #include "RAJA/RAJA.hpp"
   *
   * constexpr int CUDA_BLOCK_SIZE = 256;
   * constexpr int ASYNCHRONOUS = true;
   *
   * int size = 10000;
   * chai::UnifiedMemoryPointer<int> a(size);
   *
   * int offset = 42;
   *
   * // Both `a` and `offset` are captured by copy into the lambda expression.
   * // The execution context is not set, so the copy constructor of `a`
   * // results in a shallow copy with nothing done related to coherence
   * // management. RAJA then calls the CHAI plugin, which sets the current
   * // execution context. At that point, the lambda expression is then copied,
   * // which again triggers the copy constructor of `a`. This time, the data
   * // is made coherent on the device (which is essentially a no-op because it
   * // has not been accessed in any other execution context yet). After this,
   * // RAJA calls the CHAI plugin to reset the execution context and launches
   * // a CUDA kernel that executes the lambda expression for each index in
   * // [0, size).
   * RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE, ASYNCHRONOUS>>(RAJA::TypedRangeSegment<int>(0, size), [=] __device__ (int i) {
   *   a[i] = offset + i; 
   * });
   *
   * // The same process as described above happens, except that now `a` is
   * // made coherent on the host by synchronizing the device and the lambda
   * // expression is evaluated on the host.
   * RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, size), [=] __device__ (int i) {
   *   a[i] = i - offset;
   * });
   * \endcode
   */
  template <typename T>
  class UnifiedMemoryPointer {
    public:
      /*!
       * \brief Constructs an empty array.
       */
      constexpr UnifiedMemoryPointer() noexcept = default;

      explicit UnifiedMemoryPointer(const umpire::Allocator& allocator)
        : m_manager{new UnifiedMemoryManager(allocator)}
      {
      }

      UnifiedMemoryPointer(std::size_t size, const umpire::Allocator& allocator)
        : m_size{size},
          m_manager{new UnifiedMemoryManager(size * sizeof(T), allocator)}
      {
      }

      explicit UnifiedMemoryPointer(int allocatorID)
        : m_manager{new UnifiedMemoryManager(allocatorID)}
      {
      }

      UnifiedMemoryPointer(std::size_t size, int allocatorID)
        : m_size{size},
          m_manager{new UnifiedMemoryManager(size * sizeof(T), allocatorID)}
      {
      }

      explicit UnifiedMemoryPointer(UnifiedMemoryManager* manager)
        : m_manager{manager}
      {
        if (m_manager)
        {
          m_size = m_manager->size() / sizeof(T);
        }
      }

      /*!
       * \brief Constructs a shallow copy of an array from another and makes
       *        the data coherent in the current execution space.
       *
       * \param other The other array.
       *
       * \note This is a shallow copy.
       */
      CHAI_HOST_DEVICE UnifiedMemoryPointer(const UnifiedMemoryPointer& other) :
        m_data{other.m_data},
        m_size{other.m_size}
#if !defined(CHAI_DEVICE_COMPILE)
        , m_manager{other.m_manager}
#endif
      {
        update();
      }

      /*!
       * \brief Sets the array manager for this UnifiedMemoryPointer.
       *
       * \param manager The new array manager to be set.
       *
       * \post The UnifiedMemoryPointer takes ownership of the new manager objet.
       */
      void setManager(UnifiedMemoryManager* manager)
      {
        delete m_manager;
        m_manager = manager;

        if (m_manager)
        {
          m_size = m_manager->size() / sizeof(T);
        }
      }

      /*!
       * \brief Get the array manager associated with this UnifiedMemoryPointer.
       *
       * \return A pointer to the array manager.
       */
      UnifiedMemoryManager* getManager() const
      {
        return m_manager;
      }

      /*!
       * \brief Resizes the array to the specified new size.
       *
       * \param newSize The new size to resize the array to.
       *
       * \note This method updates the size of the array and triggers a resize operation in the array manager if it exists.
       *       If no array manager is associated, an exception is thrown.
       */
      void resize(std::size_t newSize)
      {
        if (m_manager) {
          m_size = newSize;
          m_manager->resize(newSize * sizeof(T));
        }
        else {
          m_manager = new UnifiedMemoryManager(newSize * sizeof(T));
        }
      }

      /*!
       * \brief Frees the resources associated with this array.
       *
       * \note Once free has been called, it is invalid to use any other copies
       *       of this array (since copies are shallow).
       */
      void free() {
        m_data = nullptr;
        m_size = 0;
        delete m_manager;
        m_manager = nullptr;
      }

      /*!
       * \brief Get the number of elements in the array.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE std::size_t size() const
      {
        return m_size;
      }

      CHAI_HOST_DEVICE update() const
      {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(!std::is_const<T>::value));
          // m_size = m_manager->size() / sizeof(T);
        }
#endif
      }

      /*!
       * \brief Get a pointer to the element data in the current execution context.
       *
       * \return A pointer to the element data in the current execution context.
       */
      CHAI_HOST_DEVICE T* data() const
      {
        update();
        return m_data;
      }

      /*!
       * \brief Get a const pointer to the element data in the current execution context.
       *
       * \return A const pointer to the element data in the current execution context.
       */
      CHAI_HOST_DEVICE const T* cdata() const {
#if !defined(CHAI_DEVICE_COMPILE)
        if (m_manager) {
          m_data = static_cast<T*>(m_manager->data(false));
          m_size = m_manager->size() / sizeof(T);
        }
#endif
        return m_data;
      }

      /*!
       * \brief Get the ith element in the array.
       *
       * \param i The index of the element to retrieve.
       *
       * \pre The copy constructor has been called with the execution space
       *      set to CPU or GPU (e.g. by the RAJA plugin).
       */
      CHAI_HOST_DEVICE T& operator[](std::size_t i) const {
        return m_data[i];
      }

      /*!
       * \brief Get the value of the element at the specified index.
       *
       * \param i The index of the element to retrieve.
       *
       * \return The value of the element at the specified index.
       *
       * \throw std::runtime_exception if unable to retrieve the element.
       */
      T get(std::size_t i) const {
        if (m_manager) {
          return m_manager->get(i);
        }
        else {
          throw std::out_of_range();
        }
      }

      /*!
       * \brief Set a value at a specified index in the array.
       *
       * \param i The index where the value is to be set.
       * \param value The value to set at the specified index.
       *
       * \throw std::runtime_exception if the array manager is not associated with the UnifiedMemoryPointer.
       */
      void set(std::size_t i, const T& value) {
        if (m_manager) {
          m_manager->set(i, value);
        }
        else {
          throw std::out_of_range();
        }
      }

    private:
      /*!
       * The array that is coherent in the current execution space.
       */
      T* m_data{nullptr};

      /*!
       * The number of elements in the array.
       */
      std::size_t m_size{0};

      /*!
       * The array manager controls the coherence of the array.
       */
      UnifiedMemoryManager* m_manager{nullptr};
  };  // class UnifiedMemoryPointer
}  // namespace expt
}  // namespace chai

#endif  // CHAI_UNIFIED_MEMORY_POINTER_HPP
