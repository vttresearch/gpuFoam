#pragma once

#include "traits.hpp"
#include "range.hpp"
#include "constant_range.hpp"
namespace topaz{



template <class T1, class T2>
inline CUDA_HOSTDEV auto determine_size(const T1&, const T2& rhs)
    -> std::enable_if_t<IsScalar_v<T1>, typename T2::difference_type> {
    return adl_size(rhs);
}
template <class T1, class T2>
inline CUDA_HOSTDEV auto determine_size(const T1& lhs, const T2&)
    -> std::enable_if_t<IsScalar_v<T2>, typename T1::difference_type> {
    return adl_size(lhs);
}

template <class T1,
          class T2,
          typename = std::enable_if_t<BothRangesOrNumericArrays_v<T1, T2>>>
inline CUDA_HOSTDEV auto determine_size(const T1& lhs, const T2&) {
    return adl_size(lhs);
}

template <class Range_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<Range_t>>>
inline CUDA_HOSTDEV auto rangify(Range_t& rng, Size n) {
    return take(rng, n);
}

template <class Range_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<Range_t>>>
inline CUDA_HOSTDEV auto rangify(const Range_t& rng, Size n) {
    return take(rng, n);
}

template <class Scalar,
          class Size,
          std::enable_if_t<IsScalar_v<Scalar>, bool> = true>
inline CUDA_HOSTDEV auto rangify(const Scalar& s, Size n) {
    return make_constant_range<Scalar, Size>(s, n);
}

template <class T1, class T2, class BinaryOp>
inline CUDA_HOSTDEV auto
smart_transform(const T1& lhs, const T2& rhs, BinaryOp f) {
    auto size = determine_size(lhs, rhs);
    return transform(rangify(lhs, size), rangify(rhs, size), f);
}


}