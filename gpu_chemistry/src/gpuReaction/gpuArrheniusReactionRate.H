#pragma once

#include "gpu_constants.H"
#include "device_array.H"
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
        (void)c;
        return computeAk(p, T);
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
        (void)c;
        return computeAk(p, T) * (beta_ + Ta_ / T) / T;
    }

    ///
    ///@brief Determines if the rate is a function of concentration.
    ///
    ///@return False.
    ///
    inline CUDA_HOSTDEV bool hasDdc() const { return false; }

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
                                 mdspan<gScalar, 1> ddc) const {}

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
        this->ddc(p, T, c, ddc);

        const auto Ak = computeAk(p, T);
        const auto dT = Ak * (beta_ + Ta_ / T) / T;
        return device_array<gScalar, 2>({Ak, dT});
    }
};

} // namespace FoamGpu
