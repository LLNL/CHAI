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

#include "gtest/gtest.h"

#include "chai/config.hpp"
#include "chai/ExecutionSpaces.hpp"

TEST(ExecutionSpace, Platforms)
{
  ASSERT_TRUE(chai::CPU == camp::resources::Platform::host);
  ASSERT_FALSE(chai::CPU == camp::resources::Platform::undefined);
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  ASSERT_TRUE(chai::GPU == camp::resources::Platform::cuda);
  ASSERT_TRUE(chai::GPU == camp::resources::Platform::hip);
  ASSERT_FALSE(chai::GPU == camp::resources::Platform::undefined);
#endif
}

TEST(ExecutionSpace, Host)
{
  camp::resources::Resource res{camp::resources::Host()};
  ASSERT_TRUE( chai::CPU == res.get<camp::resources::Host>().get_platform() );
}

#if defined(CHAI_ENABLE_CUDA)
TEST(ExecutionSpace, Cuda)
{
  camp::resources::Resource res{camp::resources::Cuda()};
  ASSERT_TRUE( chai::GPU == res.get<camp::resources::Cuda>().get_platform() );
}
#endif // #if defined(CHAI_ENABLE_CUDA)

#if defined(CHAI_ENABLE_HIP)
TEST(ExecutionSpace, Hip)
{
  camp::resources::Resource res{camp::resources::Hip()};
  ASSERT_TRUE( chai::GPU == res.get<camp::resources::Hip>().get_platform() );
}
#endif // #if defined(CHAI_ENABLE_CUDA)
