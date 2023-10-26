#pragma once

#include "gpuThirdBodyEfficiencies.H"

namespace FoamGpu {

template <class ReactionRate, class FallOffFunction>
class gpuFallOffReactionRate {
    // Private Data

    ReactionRate k0_;

    ReactionRate kInf_;

    FallOffFunction F_;

    gpuThirdBodyEfficiencies thirdBodyEfficiencies_;

public:
    // Constructors

    gpuFallOffReactionRate() = default;

    //- Construct from components
    inline gpuFallOffReactionRate(const ReactionRate&             k0,
                                  const ReactionRate&             kInf,
                                  const FallOffFunction&          F,
                                  const gpuThirdBodyEfficiencies& tbes)
        : k0_(k0)
        , kInf_(kInf)
        , F_(F)
        , thirdBodyEfficiencies_(tbes) {}

    //- Return the rate
    inline CUDA_HOSTDEV gScalar operator()(const gScalar                  p,
                                           const gScalar                  T,
                                           const mdspan<const gScalar, 1> c) const {
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        return kInf * (Pr / (1.0 + Pr)) * F;
    }

    //- The derivative of the rate w.r.t. temperature
    inline CUDA_HOSTDEV gScalar ddT(const gScalar                  p,
                                    const gScalar                  T,
                                    const mdspan<const gScalar, 1> c) const {
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        const gScalar dkInfdT = kInf_.ddT(p, T, c);

        return dkInfdT * (Pr / (1.0 + Pr)) * F;
    }

    //- Is the rate a function of concentration?
    inline CUDA_HOSTDEV bool hasDdc() const { return true; }

    //- The derivative of the rate w.r.t. concentration
    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1>             ddc) const {

        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        kInf_.ddc(p, T, c, ddc);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] *= (Pr / (1.0 + Pr)) * F;
        }
    }
};

} // namespace FoamGpu
