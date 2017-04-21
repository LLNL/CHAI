#ifndef CHAI_ChaiMacros_HPP
#define CHAI_ChaiMacros_HPP

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__

#define CHAI_INLINE inline

#define CHAI_UNUSED_ARG(X) 

#ifdef DEBUG
#define CHAI_LOG(file, msg) \
  std::cerr << "[" << file << "] " << msg << std::endl;
#else
#define CHAI_LOG(file, msg) 
#endif

#endif // CHAI_ChaiMacros_HPP
