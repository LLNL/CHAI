//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_pluginLinker_HPP
#define CHAI_pluginLinker_HPP

#include "chai/RajaExecutionSpacePlugin.hpp"

namespace {
  namespace anonymous_chai {
    struct pluginLinker {
      pluginLinker() {
        (void) chai::linkRajaPlugin();
      }
    } pluginLinker;
  }
}

#endif // CHAI_pluginLinker_HPP
