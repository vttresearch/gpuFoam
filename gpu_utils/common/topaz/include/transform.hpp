#pragma once

#include "traits.hpp"
#include "transform_range.hpp"
#include "zip.hpp"

namespace topaz {

template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto transform(Range_t& rng, UnaryOp f) {
    return make_transform_range(rng, f);
}

template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto transform(const Range_t& rng, UnaryOp f) {
    return make_transform_range(rng, f);
}

template <typename BinaryOp>
struct ApplyBinaryOp {

    BinaryOp op;

    inline CUDA_HOSTDEV ApplyBinaryOp(BinaryOp f)
        : op(f) {}

    template <typename Tuple>
    inline CUDA_HOSTDEV auto operator()(const Tuple& t) const {

        return op(get<0>(t), get<1>(t));
    }
};

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto transform(Range1_t& rng1, Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(Range1_t& rng1, const Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(const Range1_t& rng1, Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(const Range1_t& rng1, const Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}






} // namespace topaz