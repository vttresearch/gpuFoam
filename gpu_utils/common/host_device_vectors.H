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
