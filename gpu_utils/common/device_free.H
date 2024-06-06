#pragma once

#include "error_handling.H"

#ifdef __NVIDIA_BACKEND__
template<class T>
static inline void device_free(T* ptr){
    gpuErrorCheck(cudaFree(ptr));
}
#elifdef __AMD_BACKEND__
template<class T>
static inline void device_free(T* ptr){
    gpuErrorCheck(hipFree(ptr));
}
#endif