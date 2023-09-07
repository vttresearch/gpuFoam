#pragma once

#include "range.hpp"

#ifdef __NVIDIA_COMPILER__
#include <thrust/async/copy.h>
#include <thrust/copy.h>
#else
#include <algorithm>
#endif

namespace topaz {

#ifdef __NVIDIA_COMPILER__

template <class Range1_t, class Range2_t>
void copy(const Range1_t& src, Range2_t& dst) {
    thrust::copy(src.begin(), src.end(), dst.begin());
}

#else

template <class Range1_t, class Range2_t>
void copy(const Range1_t& src, Range2_t& dst) {
    std::copy(src.begin(), src.end(), dst.begin());
}

#endif

} // namespace topaz