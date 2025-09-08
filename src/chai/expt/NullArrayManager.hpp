//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_NULL_ARRAY_MANAGER_HPP
#define CHAI_NULL_ARRAY_MANAGER_HPP

#include "chai/expt/ArrayManager.hpp"
#include <stdexcept>

namespace chai {
namespace expt {

/*!
 * \class NullArrayManager
 *
 * \brief A null implementation of ArrayManager that doesn't actually store data.
 *
 * This class implements the ArrayManager interface but doesn't actually
 * store any data. It's implemented as a singleton.
 */
template <typename T>
class NullArrayManager : public ArrayManager<T> {
public:
  /*!
   * \brief Get the singleton instance of NullArrayManager.
   *
   * \return Reference to the singleton instance.
   */
  static NullArrayManager<T>& getInstance() {
    static NullArrayManager<T> instance;
    return instance;
  }

  /*!
   * \brief Virtual destructor.
   */
  virtual ~NullArrayManager() = default;

  /*!
   * \brief Creates a clone of this NullArrayManager.
   *
   * \return Pointer to the singleton instance.
   */
  virtual ArrayManager<T>* clone() const override {
    return &getInstance();
  }

  /*!
   * \brief Throws an exception when attempting to resize.
   *
   * \param newSize The new size (ignored).
   * \throws std::runtime_error Always throws this exception.
   */
  virtual void resize(std::size_t newSize) override {
    throw std::runtime_error("Cannot resize NullArrayManager");
  }

  /*!
   * \brief Returns 0 as the size.
   *
   * \return Always returns 0.
   */
  virtual std::size_t size() const override {
    return 0;
  }

  /*!
   * \brief Returns a nullptr for data access.
   *
   * \param context The execution context (ignored).
   * \param touch Whether to mark data as touched (ignored).
   * \return Always returns nullptr.
   */
  virtual T* data(Context context, bool touch) override {
    return nullptr;
  }

  /*!
   * \brief Throws an exception when attempting to get a value.
   *
   * \param i The index (ignored).
   * \return Never returns.
   * \throws std::runtime_error Always throws this exception.
   */
  virtual T get(std::size_t i) const override {
    throw std::runtime_error("Cannot get value from NullArrayManager");
  }

  /*!
   * \brief Throws an exception when attempting to set a value.
   *
   * \param i The index (ignored).
   * \param value The value (ignored).
   * \throws std::runtime_error Always throws this exception.
   */
  virtual void set(std::size_t i, const T& value) override {
    throw std::runtime_error("Cannot set value in NullArrayManager");
  }

private:
  /*!
   * \brief Private constructor for singleton pattern.
   */
  NullArrayManager() {}

  /*!
   * \brief Delete copy constructor.
   */
  NullArrayManager(const NullArrayManager&) = delete;
  
  /*!
   * \brief Delete assignment operator.
   */
  NullArrayManager& operator=(const NullArrayManager&) = delete;
};

}  // namespace expt
}  // namespace chai

#endif  // CHAI_NULL_ARRAY_MANAGER_HPP