#ifndef CHAI_MANAGER_HPP
#define CHAI_MANAGER_HPP

#include "chai/expt/ExecutionContext.hpp"

namespace chai {
namespace expt {
  /*!
   * \class Manager
   *
   * \brief Controls the coherence of an array.
   */
  class Manager {
    public:
      /*!
       * \brief Virtual destructor.
       */
      virtual ~Manager() = default;

      /*!
       * \brief Updates the data to be coherent in the current execution context.
       *
       * \param data [out] A coherent array in the current execution context.
       */
      virtual void update(void*& data) = 0;

      /*!
       * \brief Returns a modifiable reference to the current execution context.
       */
      static ExecutionContext& execution_context();
  };  // class Manager
}  // namespace expt
}  // namespace chai

#endif  // CHAI_MANAGER_HPP
