#pragma once

#include "gpuArrheniusReactionRate.H"
#include "gpuThirdBodyEfficiencies.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

namespace FoamGpu {

class gpuThirdBodyArrheniusReactionRate {

    gpuArrheniusReactionRate k_;
    gpuThirdBodyEfficiencies thirdBodyEfficiencies_;

public:
    gpuThirdBodyArrheniusReactionRate() = default;

    inline gpuThirdBodyArrheniusReactionRate(
        const gScalar                   A,
        const gScalar                   beta,
        const gScalar                   Ta,
        const gpuThirdBodyEfficiencies& tbes)
        : k_(A, beta, Ta)
        , thirdBodyEfficiencies_(tbes) {}

    inline gpuThirdBodyArrheniusReactionRate(
        const gpuArrheniusReactionRate& arhenius,
        const gpuThirdBodyEfficiencies& tbes)
        : k_(arhenius)
        , thirdBodyEfficiencies_(tbes) {}

    ///
    ///@brief Computes the the reaction rate.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@return gScalar The reaction rate.
    ///
    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        return thirdBodyEfficiencies_.M(c) * k_.operator()(p, T, c);
    }

    ///
    ///@brief Computes the derivative of the reaction rate w.r.t
    /// temperature.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration
    ///@return gScalar Derivative of the reaction rate w.r.t
    /// temperature.
    ///
    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {
        return thirdBodyEfficiencies_.M(c) * k_.ddT(p, T, c);
    }

    ///
    ///@brief Determines if the rate is a function of concentration.
    ///
    ///@return True.
    ///
    inline CUDA_HOSTDEV bool hasDdc() const { return true; }

    ///
    ///@brief Compute the derivative of the rate w.r.t concentration.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@param ddc A span of at least size nSpcecie where the
    /// derivative is put.
    ///
    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1> ddc) const {
        const gScalar k = k_.operator()(p, T, c);

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
        }
    }

    ///
    ///@brief Computes the reaction rate, the derivative of reaction
    /// rate w.r.t temperature and concentration. This function allows
    /// for re-use of temporaries for the different fields in the
    /// Jacobian calculation.
    ///
    ///@param p Pressure.
    ///@param T Temperature.
    ///@param c Concentration.
    ///@param ddc A span of at least size nSpcecie where the
    /// derivative is put.
    ///@return device_array<gScalar, 2> A pair of scalars where the
    /// first value is the reaction rate and second value is the
    /// derivative w.r.t temperature.
    ///
    inline CUDA_HOSTDEV device_array<gScalar, 2>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             ddc) const {

        auto          pair = k_.everything(p, T, c, ddc);
        const gScalar k    = pair[0];
        const gScalar ddT  = pair[1];

        for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
            ddc[i] = thirdBodyEfficiencies_.dMdc(c)[i] * k;
        }

        const gScalar M = thirdBodyEfficiencies_.M(c);
        return device_array<gScalar, 2>({M * k, M * ddT});
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * //

} // namespace FoamGpu
