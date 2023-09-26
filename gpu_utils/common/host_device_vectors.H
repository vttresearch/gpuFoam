#pragma once

#include <vector>
#include "cuda_host_dev.H"

#ifdef __NVIDIA_COMPILER__

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

template<class T>
using host_vector = thrust::host_vector<T>;

template<class T>
using device_vector = thrust::device_vector<T>;



#else

template<class T>
using host_vector = std::vector<T>;

template<class T>
using device_vector = std::vector<T>;


#endif


template<class Container>
static inline auto toDeviceVector(const Container& c)
{
    using value = typename Container::value_type;
    host_vector<value> temp(c.begin(), c.end());
    return device_vector<value>(temp.begin(), temp.end());
}

template<class Container>
static inline auto toStdVector(const Container& c)
{
    using value = typename Container::value_type;
    host_vector<value> temp(c.begin(), c.end());
    return std::vector<value>(temp.begin(), temp.end());
}
