#ifndef CHAI_HOST_MANAGER_HPP
#define CHAI_HOST_MANAGER_HPP

#include "chai/expt/Manager.hpp"

namespace chai {
namespace expt {
  /*!
   * \class HostManager
   *
   * \brief Controls the coherence of an array on the CPU.
   */
  class HostManager : public Manager {
    public:
      /*!
       * \brief Constructs a host array manager.
       */
      HostManager(int allocatorID, std::size_t size);

      /*!
       * \brief Copy constructor is deleted.
       */
      HostManager(const HostManager&) = delete;

      /*!
       * \brief Copy assignment operator is deleted.
       */
      HostManager& operator=(const HostManager&) = delete;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~HostManager();

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
       * \brief Get the allocator ID.
       */
      int getAllocatorID() const;

    private:
      int m_allocator_id{-1};
      std::size_t m_size{0};
      void* m_data{nullptr};
  };  // class HostManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_HOST_MANAGER_HPP
