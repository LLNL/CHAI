#ifndef CHAI_ARRAY_MANAGER_HPP
#define CHAI_ARRAY_MANAGER_HPP

namespace chai {
  /*!
   * \class ArrayManager
   *
   * \brief Controls the coherence of an array.
   *
   * \tparam T The type of element in the array.
   */
  template <typename T>
  class ArrayManager {
    public:
      /*!
       * \brief Virtual destructor.
       */
      virtual ~ArrayManager() = default;

      /*!
       * \brief Updates the size and data to be coherent in the current
       *        execution space.
       *
       * \param size [out] The number of elements in the coherent array.
       * \param data [out] A coherent array in the current execution space.
       */
      virtual void update(size_t& size, T*& data) = 0;
  };  // class ArrayManager
}  // namespace chai

#endif  // CHAI_ARRAY_MANAGER_HPP
