#ifndef CHAI_PINNED_ARRAY_VIEW_HPP
#define CHAI_PINNED_ARRAY_VIEW_HPP

#include "chai/expt/PinnedArrayContainer.hpp"

namespace chai {
namespace expt {
  /*!
   * \class PinnedArrayView
   *
   * \brief Provides a non-owning view into a PinnedArrayContainer.
   */
  template <typename T>
  class PinnedArrayView {
    public:
      /*!
       * \brief Default constructor.
       */
      PinnedArrayView() = default;

      /*!
       * \brief Construct a view from a PinnedArrayContainer.
       *
       * \param container The container to view.
       */
      explicit PinnedArrayView(PinnedArrayContainer<T>& container) :
        m_container(&container)
      {
      }

      /*!
       * \brief Get the number of elements.
       */
      size_t size() const {
        return m_container ? m_container->size() : 0;
      }

      /*!
       * \brief Get pointer to the data for the given execution context.
       */
      T* data(ExecutionContext executionContext) {
        return m_container ? m_container->data(executionContext) : nullptr;
      }

      /*!
       * \brief Get const pointer to the data for the given execution context.
       */
      const T* data(ExecutionContext executionContext) const {
        return m_container ? m_container->data(executionContext) : nullptr;
      }

      /*!
       * \brief Get element at index i for the given execution context.
       */
      T& get(ExecutionContext executionContext, size_t i) {
        return m_container->get(executionContext, i);
      }

      /*!
       * \brief Get const element at index i for the given execution context.
       */
      const T& get(ExecutionContext executionContext, size_t i) const {
        return m_container->get(executionContext, i);
      }

    private:
      PinnedArrayContainer<T>* m_container{nullptr};
  };  // class PinnedArrayView

}  // namespace expt
}  // namespace chai

#endif  // CHAI_PINNED_ARRAY_VIEW_HPP