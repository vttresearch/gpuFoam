#pragma once

#include <vector>
#include <array>
#include "error_handling.H"

#include "host_device_vectors.H"

#include "gpuMemoryResource.H"
#include "gpuThermo.H"
#include "FixedList.H"


#ifdef __NVIDIA_COMPILER__
#include <thrust/device_malloc_allocator.h>
using labelAllocator = thrust::device_malloc_allocator<gLabel>;
using scalarAllocator = thrust::device_malloc_allocator<gScalar>;

#else

using labelAllocator = std::allocator<gLabel>;
using scalarAllocator = std::allocator<gScalar>;

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
constexpr double errorTol = 1E-10;
#endif

#ifdef __NVIDIA_COMPILER__


template<class T, class R>
__global__ void on_device(T t, R* r)
{
    *r = t();
}


template<class T>
static inline Foam::scalar eval(T t)
{
    using namespace Foam;

    scalar *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(scalar)));
    on_device<<<1,1>>>(t, d_result);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    scalar h_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(scalar), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaFree(d_result));
    cudaDeviceSynchronize();
    return h_result;

}

#else

template<class T>
static inline Foam::scalar eval(T t)
{
    return t();
}
#endif


using memoryResource_t = FoamGpu::gpuMemoryResource<labelAllocator, scalarAllocator>;


template<class T, unsigned N>
static inline auto toArray(Foam::FixedList<T, N> a)
{

    std::array<T, N> ret{};
    for (size_t i = 0; i < N; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}



static inline auto toArray(typename FoamGpu::gpuThermo::coeffArray a)
{
    std::array<double, 7> ret{};
    for (size_t i = 0; i < 7; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}

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











