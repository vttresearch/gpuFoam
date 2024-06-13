#pragma once

#include <vector>
#include "cuda_host_dev.H"



template<class T>
using host_vector = std::vector<T>;

#ifdef __USING_GPU__

#include "device_vector.H"

//template<class T>
//using device_vector = device_vector<T>;

#else

template<class T>
using device_vector = std::vector<T>;


#endif


template<class Container>
static inline auto toDeviceVector(const Container& c)
{
    using value = typename Container::value_type;
    return device_vector<value>(c.begin(), c.end());
}

template<class T>
static inline std::vector<T> toStdVector(const std::vector<T>& v)
{
    return v;

}


/*
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
*/