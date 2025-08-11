#ifndef CHAI_MANAGER_HPP
#define CHAI_MANAGER_HPP

#include <cstddef>

namespace chai {
namespace expt {
  /*!
   * \class Manager
   *
   * \brief Controls the coherence of an array.
   */
  class Manager {
    public:
      using size_type = std::size_t;

      /*!
       * \brief Virtual destructor.
       */
      virtual ~Manager() = default;

      /*!
       * \brief Resize the contained array.
       */
      virtual void resize(size_type newSize) = 0;

      /*!
       * \brief Updates the data to be coherent in the current execution context.
       *
       * \param data [out] A coherent array in the current execution context.
       */
      virtual void* data(bool touch) = 0;
  };  // class Manager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_MANAGER_HPP
