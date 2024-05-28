#pragma once

#include "error_handling.H"

template<class T>
static inline void device_free(T* ptr){
    gpuErrorCheck(cudaFree(ptr));
}