#pragma once

#include "cuda_host_dev.H"
#include <iostream>


#ifdef __USING_GPU__
    #ifdef __NVIDIA_BACKEND__
    #include "cuda_runtime.h"

    #define gpuErrorCheck(call) \
    do{ \
        cudaError_t gpuErr = call; \
        if(cudaSuccess != gpuErr){ \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr)); \
        exit(1); \
        } \
    }while(0)


    #else //AMD
    #include<hip/hip_runtime.h>
    #define gpuErrorCheck(call) \
    do{ \
        hipError_t gpuErr = call; \
        if(hipSuccess != gpuErr){ \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1); \
        } \
    }while(0)
    #endif

#else
    #define gpuErrorCheck(val)
#endif
