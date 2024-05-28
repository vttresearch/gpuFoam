#pragma once

#include "error_handling.H"

template<class T>
static inline T* device_allocate(size_t length){

    T*         ptr;
    const auto bytesize = length * sizeof(T);
    gpuErrorCheck(cudaMalloc((void**)&ptr, bytesize));
    return ptr;
}