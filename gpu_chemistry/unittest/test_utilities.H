#pragma once

#include <vector>
#include <array>
#include "error_handling.H"

#include "host_device_vectors.H"


#include "gpuThermo.H"
#include "FixedList.H"






constexpr int CH4_idx = 0;
constexpr int O2_idx = 4;
constexpr int N2_idx = 47;

constexpr double CH4_val = 0.2;
constexpr double O2_val = 0.5;
constexpr double N2_val = 0.3;
constexpr double T_inf = 1000.0;
constexpr double p_inf = 1.36789e+06;

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


template<class T>
static inline auto to_std_vec(const T& c)
{
    using value_type = typename T::value_type;
    host_vector<value_type> temp(c.begin(), c.end());
    return std::vector<value_type>(temp.begin(), temp.end());
}

template<class T>
static inline auto to_device_vec(const T& c)
{
    using value_type = typename T::value_type;
    host_vector<value_type> temp(c.begin(), c.end());
    return device_vector<value_type>(temp.begin(), temp.end());
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
template<class T>
static inline void assign_test_condition(T& f)
{
    gLabel nSpecie = 53;
    if (gLabel(f.size()) != nSpecie + 2){
        throw std::logic_error("Field size not gri");
    }

    for (auto& e : f)
    {
        e = 0.0;
    }
    f[CH4_idx] = CH4_val;
    f[O2_idx] = O2_val;
    f[N2_idx] = N2_val;

    f[nSpecie] = T_inf;
    f[nSpecie+1] = p_inf;
}











