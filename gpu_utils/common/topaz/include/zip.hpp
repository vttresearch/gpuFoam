#pragma once

#include "traits.hpp"
#include "zip_range.hpp"

namespace topaz {

template <class Range_t>
inline CUDA_HOSTDEV auto zip(Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class Range_t>
inline CUDA_HOSTDEV auto zip(const Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class Range1_t, class Range2_t>
inline CUDA_HOSTDEV auto zip(Range1_t& rng1, Range2_t& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class Range1_t, class Range2_t>
inline CUDA_HOSTDEV auto zip(Range1_t& rng1, const Range2_t& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class Range1_t, class Range2_t>
inline CUDA_HOSTDEV auto zip(const Range1_t& rng1, Range2_t& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class Range1_t, class Range2_t>
inline CUDA_HOSTDEV auto zip(const Range1_t& rng1, const Range2_t& rng2) {
    return make_zip_range(rng1, rng2);
}
} // namespace topaz