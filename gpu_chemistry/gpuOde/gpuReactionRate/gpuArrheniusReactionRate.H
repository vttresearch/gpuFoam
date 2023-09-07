#pragma once

#include "gpuConstants.H"
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


    inline CUDA_HOSTDEV void preEvaluate() const {}

    inline CUDA_HOSTDEV void postEvaluate() const {}

    template <class Span>
    inline CUDA_HOSTDEV gScalar operator()(const gScalar p,
                                           const gScalar T,
                                           const Span&   c,
                                           const gLabel  li) const {
        (void)c;
        (void)li;
        return computeAk(p, T);
    }

    template <class Span>
    inline CUDA_HOSTDEV gScalar ddT(const gScalar p,
                                    const gScalar T,
                                    const Span&   c,
                                    const gLabel  li) const {
        (void)c;
        (void)li;
        return computeAk(p, T) * (beta_ + Ta_ / T) / T;
    }

    inline CUDA_HOSTDEV bool hasDdc() const { return false; }

    template <class Span1, class Span2>
    inline CUDA_HOSTDEV void ddc(const gScalar p,
                                 const gScalar T,
                                 const Span1&  c,
                                 const gLabel  li,
                                 Span2&        ddc) const {
        for (gLabel i = 0; i < gLabel(std::size(ddc)); ++i) { ddc[i] = 0; }
    }
};

} // namespace FoamGpu
