#pragma once

#include "cuda_host_dev.H"

#include <limits>

using gLabel  = int;
using gScalar = double;

// Check OpenFOAM/OpenFOAM-gpu/etc/controlDict SI

// Avagadro number
#define gpuNA double(6.02214e+26)

// Boltzman constant
#define gpuk double(1.38065e-23)

#define gpuRR double(gpuNA * gpuk)

// Standard pressure
#define gpuPstd double(1e5)

// Standard temperature
#define gpuTstd double(298.15)

#ifdef __USING_GPU__
    #include <cuda/std/limits>

    //::cuda::std::numeric_limits<float>::max();

    #define gpuLabelMax gLabel(::cuda::std::numeric_limits<gLabel>::max() / 10)

    #define gpuVGreat double(::cuda::std::numeric_limits<double>::max() / 10.0)

    #define gpuVSmall double(::cuda::std::numeric_limits<double>::min())

    #define gpuSmall double(::cuda::std::numeric_limits<double>::epsilon())

    #define gpuScalarNaN double(::cuda::std::numeric_limits<double>::signaling_NaN())

#else

    #define gpuLabelMax gLabel(std::numeric_limits<gLabel>::max() / 10)

    #define gpuVGreat double(std::numeric_limits<double>::max() / 10.0)

    #define gpuVSmall double(std::numeric_limits<double>::min())

    #define gpuSmall double(std::numeric_limits<double>::epsilon())

    #define gpuScalarNaN double(std::numeric_limits<double>::signaling_NaN())


#endif




#define gpuGreat double(1.0 / gpuSmall)

#define gpuLogTen double(2.302585092994046)

