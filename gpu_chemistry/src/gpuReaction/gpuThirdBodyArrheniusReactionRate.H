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

    inline CUDA_HOSTDEV gScalar operator()(const gScalar                  p,
                                           const gScalar                  T,
                                           const mdspan<const gScalar, 1> c) const {
        return thirdBodyEfficiencies_.M(c) *
               gpuArrheniusReactionRate::operator()(p, T, c);
    }

    //- The derivative of the rate w.r.t. temperature
    inline CUDA_HOSTDEV gScalar ddT(const gScalar                  p,
                                    const gScalar                  T,
                                    const mdspan<const gScalar, 1> c) const {
        return thirdBodyEfficiencies_.M(c) *
               gpuArrheniusReactionRate::ddT(p, T, c);
    }

    //- Is the rate a function of concentration?
    inline CUDA_HOSTDEV bool hasDdc() const { return true; }

    //- The derivative of the rate w.r.t. concentration
    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1>             ddc) const {
        const gScalar k = gpuArrheniusReactionRate::operator()(p, T, c);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
        }
    }

    inline CUDA_HOSTDEV std::pair<gScalar, gScalar>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             ddc) const {

        auto pair = gpuArrheniusReactionRate::everything(p, T, c, ddc);
        const gScalar k = std::get<0>(pair);
        const gScalar ddT = std::get<1>(pair);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
        }

        const gScalar M = thirdBodyEfficiencies_.M(c);
        return std::make_pair(M * k, M * ddT);

    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // namespace FoamGpu
