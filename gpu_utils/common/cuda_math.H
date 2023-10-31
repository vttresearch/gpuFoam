#pragma once


#ifndef __NVIDIA_COMPILER__
#include <cmath>

static inline double abs(const double& d) {return std::abs(d);}

//NOTE! These are host only
static inline const double& max(const double& lhs, const double& rhs)
{
    return std::max(lhs, rhs);
}

static inline const double& min(const double& lhs, const double& rhs)
{
    return std::min(lhs, rhs);
}

static inline auto adl_isfinite(double d)
{
    return std::isfinite(d);
}

#else

static inline CUDA_HOSTDEV auto adl_isfinite(double d)
{
    return isfinite(d);
}

#endif



//CUDA math library has some optimized versions of these