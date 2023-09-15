#pragma once

#include "gpuArrheniusReactionRate.H"
#include "gpuThirdBodyEfficiencies.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu {

class gpuThirdBodyArrheniusReactionRate : gpuArrheniusReactionRate {

    gpuThirdBodyEfficiencies thirdBodyEfficiencies_;

public:
    // Constructors
    gpuThirdBodyArrheniusReactionRate() = default;

    //- Construct from components
    inline gpuThirdBodyArrheniusReactionRate(
        const gScalar                   A,
        const gScalar                   beta,
        const gScalar                   Ta,
        const gpuThirdBodyEfficiencies& tbes)
        : gpuArrheniusReactionRate(A, beta, Ta)
        , thirdBodyEfficiencies_(tbes) {}

    //- Construct from components
    inline gpuThirdBodyArrheniusReactionRate(
        const gpuArrheniusReactionRate& arhenius,
        const gpuThirdBodyEfficiencies& tbes)
        : gpuArrheniusReactionRate(arhenius)
        , thirdBodyEfficiencies_(tbes) {}


    template <class Span>
    inline CUDA_HOSTDEV gScalar operator()(const gScalar p,
                                           const gScalar T,
                                           const Span    c,
                                           const gLabel  li) const {
        return thirdBodyEfficiencies_.M(c) *
               gpuArrheniusReactionRate::operator()(p, T, c, li);
    }

    //- The derivative of the rate w.r.t. temperature
    template <class Span>
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar p, const gScalar T, const Span c, const gLabel li) const {
        return thirdBodyEfficiencies_.M(c) *
               gpuArrheniusReactionRate::ddT(p, T, c, li);
    }

    //- Is the rate a function of concentration?
    inline CUDA_HOSTDEV bool hasDdc() const { return true; }

    //- The derivative of the rate w.r.t. concentration
    template <class Span1, class Span2>
    inline CUDA_HOSTDEV void ddc(const gScalar p,
                                 const gScalar T,
                                 const Span1&  c,
                                 const gLabel  li,
                                 Span2&        ddc) const {
        const gScalar k = gpuArrheniusReactionRate::operator()(p, T, c, li);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
        }
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // namespace FoamGpu
