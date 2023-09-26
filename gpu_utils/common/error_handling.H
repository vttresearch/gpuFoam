#pragma once

#include "cuda_host_dev.H"
#include <iostream>



#ifdef __NVIDIA_COMPILER__
#include <assert.h>
#include "cuda_runtime.h"


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
static CUDA_HOSTDEV void check(T err, const char* const func, const char* const file,
           const int line, bool abort=true)
{
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime error at: %s %s %s %d\n", cudaGetErrorString(err), file, func, line);
        if (abort) assert(0);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
static CUDA_HOSTDEV void checkLast(const char* const file, const int line, bool abort=true)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        printf("CUDA Runtime error at: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) assert(0);
    }
}


#else

#define CHECK_CUDA_ERROR(val)
#define CHECK_LAST_CUDA_ERROR()



#endif
