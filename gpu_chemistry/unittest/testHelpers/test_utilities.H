#pragma once

#include <vector>
#include <array>
#include "error_handling.H"

#include "host_device_vectors.H"

#include "gpuThermo.H"
#include "gpuMemoryResource.H"
#include "cpuMemoryResource.H"



#ifdef __NVIDIA_COMPILER__


using memoryResource_t = FoamGpu::gpuMemoryResource;

#else


using memoryResource_t = FoamGpu::cpuMemoryResource;

#endif



//Note, here the "error" is a comparison against a cpu result.
//When compiled with an nvidia compiler, the arithmetic operations are computed
//differently since nvcc and nvc++ use fused multiply add (fma) in many places.
//Therefore a slightly more tolerance is allowed when comparing gpu result agains
//cpu result. For better match use the -nofma switch. However, we want to allow
//for the compiler to do optimizations.
#ifdef __NVIDIA_COMPILER__
constexpr double errorTol = 1E-7;
#else
constexpr double errorTol = 1E-9;
#endif

#ifdef __NVIDIA_COMPILER__


template<class T, class R>
__global__ void on_device(T t, R* r)
{
    *r = t();
}


template<class T>
static inline gScalar eval(T t)
{

    gScalar *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(gScalar)));
    on_device<<<1,1>>>(t, d_result);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    gScalar h_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(gScalar), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaFree(d_result));
    cudaDeviceSynchronize();
    return h_result;

}

#else

template<class T>
static inline gScalar eval(T t)
{
    return t();
}
#endif







static inline double random_number(double LO, double HI){
    double r = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r;
}

template<class T>
static inline void fill_random(T& v, double LO = 0.0, double HI = 1.0)
{


    for (auto& e : v)
    {
        e = random_number(LO, HI);
    }

}














