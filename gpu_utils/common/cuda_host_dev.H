#pragma once

#if defined(__HIP__)
    #define __AMD_BACKEND__
#endif

#if defined(__NVCOMPILER) || defined(__NVCC__)
    #define __NVIDIA_BACKEND__
#endif

#if defined(__AMD_BACKEND__) || defined(__NVIDIA_BACKEND__)
    #define __USING_GPU__
#endif


#ifdef __USING_GPU__
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif

#ifdef __USING_GPU__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
