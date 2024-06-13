#pragma once

#include "error_handling.H"

#ifdef __USING_GPU__

    #ifdef __NVIDIA_BACKEND__
    template<class T>
    static inline void device_free(T* ptr){
        gpuErrorCheck(cudaFree(ptr));
    }
    #else //AMD
    template<class T>
    static inline void device_free(T* ptr){
        gpuErrorCheck(hipFree(ptr));
    }
#endif

#endif