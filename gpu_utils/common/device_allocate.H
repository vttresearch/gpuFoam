#pragma once

#include "error_handling.H"

#ifdef __USING_GPU__
    #ifdef __NVIDIA_BACKEND__
    template<class T>
    static inline T* device_allocate(size_t length){

        T*         ptr;
        const auto bytesize = length * sizeof(T);
        gpuErrorCheck(cudaMalloc((void**)&ptr, bytesize));
        return ptr;
    }
    #else //AMD
    template<class T>
    static inline T* device_allocate(size_t length){

        T*         ptr;
        const auto bytesize = length * sizeof(T);
        gpuErrorCheck(hipMalloc((void**)&ptr, bytesize));
        return ptr;
    }
    #endif
#endif