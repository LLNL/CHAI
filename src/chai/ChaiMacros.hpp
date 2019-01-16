// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
// rights reserved.
//
// Produced at the Lawrence Livermore National Laboratory.
//
// This file is part of CHAI.
//
// LLNL-CODE-705877
//
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#ifndef CHAI_ChaiMacros_HPP
#define CHAI_ChaiMacros_HPP

#include "chai/config.hpp"

#if defined(CHAI_ENABLE_CUDA)

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__

#else

#define CHAI_HOST
#define CHAI_DEVICE
#define CHAI_HOST_DEVICE

#endif

#define CHAI_INLINE inline

#define CHAI_UNUSED_ARG(X)

#if CHAI_DEBUG > 1
#define CHAI_LOG(file, msg) \
  std::cerr << "[" << file << "] " << msg << std::endl;
#else
#define CHAI_LOG(file, msg)
#endif

#endif  // CHAI_ChaiMacros_HPP
