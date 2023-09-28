#pragma once

#include "cuda_host_dev.H"

#ifdef __NVIDIA_COMPILER__
#include <thrust/device_ptr.h>
#endif

template<class T>
struct Caster{
    auto operator()(T p) const{
        return p;
    }
};
#ifdef __NVIDIA_COMPILER__
template<class T>
struct Caster<thrust::device_ptr<T>>{

    auto operator()(T p) const{
        return thrust::raw_pointer_cast(p);
    }

};
#endif

template<class T>
static inline
CUDA_HOSTDEV auto raw_pointer_cast(T p){
    Caster<T> c;
    return c(p);

}