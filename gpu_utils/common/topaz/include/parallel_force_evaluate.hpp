#pragma once

#include "range.hpp"

#ifdef __NVIDIA_COMPILER__
#include <thrust/transform.h>
#else
#include <algorithm>
//#include <execution>
#endif

namespace topaz {

namespace detail {

struct NoOp {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& d) const -> decltype(d) {
        return d;
    }
};
} // namespace detail

template <class Policy, class Range1_t, class Range2_t>
void parallel_force_evaluate(Policy p, const Range1_t& src, Range2_t& dst) {

#ifdef __NVIDIA_COMPILER__
    thrust::transform(p, src.begin(), src.end(), dst.begin(), detail::NoOp{});
#else
    std::copy(p, src.begin(), src.end(), dst.begin());
#endif
}

} // namespace topaz