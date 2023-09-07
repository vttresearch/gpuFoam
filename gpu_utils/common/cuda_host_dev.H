#pragma once


#if defined(__NVCOMPILER) || defined(__NVCC__)
    #define __NVIDIA_COMPILER__
#endif

#ifdef __NVIDIA_COMPILER__
#define DEVICE __device__
#else
#define DEVICE
#endif


#ifdef __NVIDIA_COMPILER__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
