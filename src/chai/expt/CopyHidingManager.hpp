#ifndef CHAI_COPY_HIDING_MANAGER_HPP
#define CHAI_COPY_HIDING_MANAGER_HPP

#include "chai/expt/Manager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class CopyHidingManager
   *
   * \brief Controls the coherence of an array on the host and device.
   */
  class CopyHidingManager : public Manager {
    public:
      /*!
       * \brief Constructs a host array manager.
       */
      CopyHidingManager(int hostAllocatorID,
                        int deviceAllocatorID,
                        std::size_t size);

      /*!
       * \brief Copy constructor is deleted.
       */
      CopyHidingManager(const CopyHidingManager&) = delete;

      /*!
       * \brief Copy assignment operator is deleted.
       */
      CopyHidingManager& operator=(const CopyHidingManager&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~CopyHidingManager();

      /*!
       * \brief Get the number of elements.
       */
      virtual std::size_t size() const override;

      /*!
       * \brief Updates the data to be coherent in the current execution space.
       *
       * \param data [out] A coherent array in the current execution space.
       */
      virtual void* data(ExecutionContext context, bool touch) override;

      /*!
       * \brief Get the host allocator ID.
       */
      int getHostAllocatorID() const;

      /*!
       * \brief Get the device allocator ID.
       */
      int getDeviceAllocatorID() const;

      /*!
       * \brief Get the last touch.
       */
      ExecutionContext getTouch() const;

    private:
      int m_host_allocator_id{-1};
      int m_device_allocator_id{-1};
      std::size_t m_size{0};
      void* m_host_data{nullptr};
      void* m_device_data{nullptr};
      ExecutionContext m_touch{ExecutionContext::NONE};
  };  // class CopyHidingManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_COPY_HIDING_MANAGER_HPP
