#pragma once

#include "cuda_host_dev.H"
#include <iostream>



#ifdef __NVIDIA_COMPILER__
#include <assert.h>
#include "cuda_runtime.h"

#define gpuErrorCheck(call) \
do{ \
    cudaError_t gpuErr = call; \
    if(cudaSuccess != gpuErr){ \
    printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr)); \
    exit(1); \
    } \
}while(0)





#else

#define gpuErrorCheck(val)



#endif
