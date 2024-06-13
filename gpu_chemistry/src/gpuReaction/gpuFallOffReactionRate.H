#pragma once

#include "gpuThirdBodyEfficiencies.H"

namespace FoamGpu {

template <class ReactionRate, class FallOffFunction>
class gpuFallOffReactionRate {

    ReactionRate k0_;

    ReactionRate kInf_;

    FallOffFunction F_;

    gpuThirdBodyEfficiencies thirdBodyEfficiencies_;

public:
    gpuFallOffReactionRate() = default;

    inline gpuFallOffReactionRate(
        const ReactionRate&             k0,
        const ReactionRate&             kInf,
        const FallOffFunction&          F,
        const gpuThirdBodyEfficiencies& tbes)
        : k0_(k0)
        , kInf_(kInf)
        , F_(F)
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
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        return kInf * (Pr / (1.0 + Pr)) * F;
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
        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        const gScalar dkInfdT = kInf_.ddT(p, T, c);

        return dkInfdT * (Pr / (1.0 + Pr)) * F;
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

        const gScalar k0   = k0_(p, T, c);
        const gScalar kInf = kInf_(p, T, c);
        const gScalar M    = thirdBodyEfficiencies_.M(c);
        const gScalar Pr   = k0 / kInf * M;
        const gScalar F    = F_(T, Pr);

        const gScalar temp    = (Pr / (1.0 + Pr)) * F;
        const gScalar k       = kInf * temp;
        const gScalar dkInfdT = kInf_.ddT(p, T, c);
        const gScalar ddT     = dkInfdT * temp;

        if (kInf_.hasDdc()) {
            kInf_.ddc(p, T, c, ddc);

            for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
                ddc[i] *= temp;
            }
        } else {
            for (gLabel i = 0; i < gLabel(ddc.size()); ++i) {
                ddc[i] = gScalar(0);
            }
        }

        return {k, ddT};

        // return device_array<gScalar, 2>({k, ddT});
    }
};

} // namespace FoamGpu
