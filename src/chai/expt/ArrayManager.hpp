#ifndef CHAI_ARRAY_MANAGER_HPP
#define CHAI_ARRAY_MANAGER_HPP

#include <cstddef>

namespace chai {
namespace expt {
  /*!
   * \class ArrayManager
   *
   * \brief Controls the coherence of an array.
   */
  template <typename ElementT>
  class ArrayManager {
    public:
      /*!
       * \brief Virtual destructor.
       */
      virtual ~ArrayManager() = default;

      /*!
       * \brief Creates a clone of this ArrayManager.
       *
       * \return A new ArrayManager object that is a clone of this instance.
       */
      virtual ArrayManager* clone() const = 0;

      /*!
       * \brief Resizes the array to the specified new size.
       *
       * \param newSize The new size to resize the array to.
       */
      virtual void resize(std::size_t newSize) = 0;

      /*!
       * \brief Returns the size of the contained array.
       *
       * \return The size of the contained array.
       */
      virtual std::size_t size() const = 0;

      /*!
       * \brief Updates the data to be coherent in the current execution context.
       *
       * \param data [out] A coherent array in the current execution context.
       */
      virtual void* data(ExecutionSpace space, bool touch) const = 0;
  };  // class ArrayManager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_ARRAY_MANAGER_HPP
