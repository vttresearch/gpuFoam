#pragma once

#include "gpu_constants.H"

namespace FoamGpu {

inline CUDA_HOSTDEV bool isInteger(gScalar N) {

    // Convert float value
    // of N to integer
    int X = N;

    double temp2 = N - X;

    // If N is not equivalent
    // to any integer
    if (temp2 > 0) { return false; }
    return true;
}

inline CUDA_HOSTDEV gScalar integerPow(const gScalar x,
                                       const gLabel  e) {
    if (e == 0) { return 1; }

    gScalar xx = e > 0 ? x : 1 / x;

    gScalar y = 1;

    for (gLabel i = e; i != 0; i /= 2) {
        if (i % 2) { y *= xx; }

        xx *= xx;
    }

    return y;
}

struct gpuSpecieExponent {
    
    gLabel noIntegerExponent_ = gpuLabelMax;

    gLabel integerExponent_;

    gScalar scalarExponent_;

    inline CUDA_HOSTDEV gpuSpecieExponent()
        : integerExponent_(gpuLabelMax)
        , scalarExponent_(gpuScalarNaN) {}

    inline CUDA_HOSTDEV
    gpuSpecieExponent(const gLabel integerExponent)
        : integerExponent_(integerExponent)
        , scalarExponent_(integerExponent) {}

    inline CUDA_HOSTDEV
    gpuSpecieExponent(const gScalar scalarExponent)
        : integerExponent_(gpuLabelMax)
        , scalarExponent_(scalarExponent) {}

    inline CUDA_HOSTDEV bool hasIntegerExponent() const {
        return integerExponent_ != noIntegerExponent_;
    }

    inline CUDA_HOSTDEV operator gScalar() const {
        return scalarExponent_;
    }

    inline CUDA_HOSTDEV gpuSpecieExponent&
                        operator=(const gLabel integerExponent) {
        return *this = gpuSpecieExponent(integerExponent);
    }

    inline CUDA_HOSTDEV gpuSpecieExponent&
                        operator=(const gScalar scalarExponent) {
        return *this = gpuSpecieExponent(scalarExponent);
    }

    inline CUDA_HOSTDEV gpuSpecieExponent operator-() const {
        if (hasIntegerExponent()) {
            return -integerExponent_;
        } else {
            return -scalarExponent_;
        }
    }

    inline CUDA_HOSTDEV friend gpuSpecieExponent
    operator+(const gpuSpecieExponent& a,
              const gpuSpecieExponent& b) {
        if (a.hasIntegerExponent() && b.hasIntegerExponent()) {
            return a.integerExponent_ + b.integerExponent_;
        } else {
            return a.scalarExponent_ + b.scalarExponent_;
        }
    }

    inline CUDA_HOSTDEV friend gpuSpecieExponent
    operator-(const gpuSpecieExponent& a,
              const gpuSpecieExponent& b) {
        return a + (-b);
    }
};

inline CUDA_HOSTDEV gScalar speciePow(const gScalar            x,
                                      const gpuSpecieExponent& e) {
    if (e.hasIntegerExponent()) {
        return integerPow(x, e.integerExponent_);
    } else {
        return pow(x, e.scalarExponent_);
    }
}

} // namespace FoamGpu
