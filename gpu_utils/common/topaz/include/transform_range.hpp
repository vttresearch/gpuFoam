#pragma once

#include "range.hpp"

#ifdef __NVIDIA_COMPILER__
#include <thrust/iterator/transform_iterator.h>
#else
#include <boost/iterator/transform_iterator.hpp>
#endif
namespace topaz {




namespace detail {

#ifdef __NVIDIA_COMPILER__

    template <class Func, class Iter>
    using transform_iterator = thrust::transform_iterator<Func, Iter>;

    template <class Func, class Iter>
    inline CUDA_HOSTDEV auto make_transform_iterator(Iter it, Func f) {
        return thrust::make_transform_iterator(it, f);
    }

#else

    template<class Func, class Iter>
    using transform_iterator = boost::transform_iterator<Func, Iter>;

    template<class Func, class Iter>
    inline CUDA_HOSTDEV auto make_transform_iterator(Iter it, Func f){
        return boost::make_transform_iterator(it, f);
    }

#endif
} // namespace detail

template <typename UnaryFunction, typename Iterator>
struct TransformRange
    : public Range<detail::transform_iterator<UnaryFunction, Iterator>> {

    using parent = Range<detail::transform_iterator<UnaryFunction, Iterator>>;

    inline CUDA_HOSTDEV
    TransformRange(Iterator first, Iterator last, UnaryFunction f)
        : parent(detail::make_transform_iterator(first, f),
                 detail::make_transform_iterator(last, f)) {}

    template <class Range_t>
    inline CUDA_HOSTDEV TransformRange(Range_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}

    template <class Range_t>
    inline CUDA_HOSTDEV TransformRange(const Range_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}
};

template <typename Function, typename Iterator>
inline CUDA_HOSTDEV auto
make_transform_range(Iterator first, Iterator last, Function f) {
    return TransformRange<Function, Iterator>(first, last, f);
}

template <typename Function, class Range_t>
inline CUDA_HOSTDEV auto make_transform_range(Range_t& rng, Function f) {
    using iterator = decltype(std::begin(rng));
    return TransformRange<Function, iterator>(rng, f);
}

template <typename Function, class Range_t>
inline CUDA_HOSTDEV auto make_transform_range(const Range_t& rng, Function f) {
    using iterator = decltype(std::begin(rng));
    return TransformRange<Function, iterator>(rng, f);
}

} // namespace topaz