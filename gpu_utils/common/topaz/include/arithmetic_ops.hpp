#pragma once

//#include <cmath>
#include <math.h>
#include "range.hpp"
#include "smart_transform.hpp"
#include "traits.hpp"

namespace topaz {

struct Plus {
    template <class T1, class T2>
    inline CUDA_HOSTDEV auto operator()(const T1& lhs, const T2& rhs) const
        -> decltype(lhs + rhs) {
        return lhs + rhs;
    }
};

struct Minus {
    template <class T1, class T2>
    inline CUDA_HOSTDEV auto operator()(const T1& lhs, const T2& rhs) const
        -> decltype(lhs - rhs) {
        return lhs - rhs;
    }
};

struct Multiplies {
    template <class T1, class T2>
    inline CUDA_HOSTDEV auto operator()(const T1& lhs, const T2& rhs) const
        -> decltype(lhs * rhs) {
        return lhs * rhs;
    }
};

struct Divides {
    template <class T1, class T2>
    inline CUDA_HOSTDEV auto operator()(const T1& lhs, const T2& rhs) const
        -> decltype(lhs / rhs) {
        return lhs / rhs;
    }
};

template<class T>
inline CUDA_HOSTDEV const T& adl_max(const T& a, const T& b)
{
    return (a < b) ? b : a;
}

struct Max {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(adl_max(lhs, rhs)) {
            return adl_max(lhs, rhs);
    }
};


template<class T>
inline CUDA_HOSTDEV const T& adl_min(const T& a, const T& b)
{
    return (b < a) ? b : a;
}

struct Min {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(adl_min(lhs, rhs)) {
            return adl_min(lhs, rhs);
    }
};

inline CUDA_HOSTDEV float  adl_sqrt(float s) { return ::sqrtf(s); }
inline CUDA_HOSTDEV double adl_sqrt(double s) { return ::sqrt(s); }
struct Sqrt {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(adl_sqrt(t)) {
        return adl_sqrt(t);
    }
};

struct Exp {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(::exp(t)) {
        return ::exp(t);
    }
};

struct Log_e {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(::log(t)) {
        return ::log(t);
    }
};

struct Pow {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& x, const T& power) const
        -> decltype(::pow(x,power)) {
        return ::pow(x, power);
    }
};


inline CUDA_HOSTDEV float  adl_erf(float s) { return ::erff(s); }
inline CUDA_HOSTDEV double adl_erf(double s) { return ::erf(s); }
struct Erf {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(adl_erf(t)) {
        return adl_erf(t);
    }
};

struct Cbrt {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(::cbrt(t)) {
        return ::cbrt(t);
    }
};




template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator+(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Plus{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator-(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Minus{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator*(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Multiplies{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator/(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Divides{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto max(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Max{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto min(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Min{});
}


template <class T, class Scalar_t, typename = std::enable_if_t<IsRangeOrNumericArray_v<T> && IsScalar_v<Scalar_t>>>
inline CUDA_HOSTDEV auto pow(const T& t, Scalar_t power) {
    return smart_transform(t, power, Pow{});
}



template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto sqr(const T& t) {
    return t * t;
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto sqrt(const T& t) {
    return transform(t, Sqrt{});
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto pow2(const T& t) {
    using value_type = typename T::value_type;
    return pow(t, value_type(2));
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto pow3(const T& t) {
    using value_type = typename T::value_type;
    return pow(t, value_type(3));
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto exp(const T& t) {
    return transform(t, Exp{});
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto log(const T& t) {
    return transform(t, Log_e{});
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto erf(const T& t) {
    return transform(t, Erf{});
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto cbrt(const T& t) {
    return transform(t, Cbrt{});
}

} // namespace topaz