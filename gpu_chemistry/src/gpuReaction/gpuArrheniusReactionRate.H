#pragma once

#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

class gpuArrheniusReactionRate {
    // Private Data

    gScalar A_;
    gScalar beta_;
    gScalar Ta_;

    inline CUDA_HOSTDEV gScalar computeAk(const gScalar p,
                                          const gScalar T) const {
        gScalar ak = A_;

        if (fabs(beta_) > gpuVSmall) { ak *= pow(T, beta_); }

        if (fabs(Ta_) > gpuVSmall) { ak *= exp(-Ta_ / T); }

        return ak;
    }

public:
    gpuArrheniusReactionRate() = default;

    inline gpuArrheniusReactionRate(const gScalar A,
                                    const gScalar beta,
                                    const gScalar Ta)
        : A_(A)
        , beta_(beta)
        , Ta_(Ta) {}

    inline CUDA_HOSTDEV gScalar operator()(const gScalar                  p,
                                           const gScalar                  T,
                                           const mdspan<const gScalar, 1> c) const {
        (void)c;
        return computeAk(p, T);
    }

    inline CUDA_HOSTDEV gScalar ddT(const gScalar                  p,
                                    const gScalar                  T,
                                    const mdspan<const gScalar, 1> c) const {
        (void)c;
        return computeAk(p, T) * (beta_ + Ta_ / T) / T;
    }

    inline CUDA_HOSTDEV bool hasDdc() const { return false; }

    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1>             ddc) const {
        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) { ddc[i] = 0; }
    }
};

} // namespace FoamGpu
