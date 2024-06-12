#pragma once

// #include <variant>
#include "variant.hpp"

#include "gpuArrheniusReactionRate.H"
#include "gpuFallOffReactionRate.H"
#include "gpuLindemannFallOffFunction.H"
#include "gpuThirdBodyArrheniusReactionRate.H"
#include "gpuTroeFallOffFunction.H"

namespace FoamGpu {

namespace gpu {

using Arrhenius          = gpuArrheniusReactionRate;
using ThirdBodyArrhenius = gpuThirdBodyArrheniusReactionRate;
using ArrheniusLindemannFallOff =
    gpuFallOffReactionRate<gpuArrheniusReactionRate,
                           gpuLindemannFallOffFunction>;
using ArrheniusTroeFallOff =
    gpuFallOffReactionRate<gpuArrheniusReactionRate,
                           gpuTroeFallOffFunction>;

using ReactionRate = variant::variant<Arrhenius,
                                      ThirdBodyArrhenius,
                                      ArrheniusLindemannFallOff,
                                      ArrheniusTroeFallOff>;

/*
using ReactionRate =
std::variant
<
    Arrhenius,
    ThirdBodyArrhenius,
    ArrheniusLindemannFallOff,
    ArrheniusTroeFallOff
>;
*/
} // namespace gpu

struct RRVisitor {

    CUDA_HOSTDEV
    RRVisitor(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc)
        : p(pp)
        , Temperature(TT)
        , c(cc) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;

    template <class T>
    CUDA_HOSTDEV gScalar operator()(const T& i) const {
        return i.operator()(p, Temperature, c);
    }
};

struct hasDdcVisitor {

    hasDdcVisitor() = default;

    template <class T>
    CUDA_HOSTDEV bool operator()(const T& i) const {
        return i.hasDdc();
    }
};

struct ddcVisitor {

    CUDA_HOSTDEV
    ddcVisitor(gScalar                  pp,
               gScalar                  TT,
               mdspan<const gScalar, 1> cc,
               mdspan<gScalar, 1>       ddkfdc)
        : p(pp)
        , Temperature(TT)
        , c(cc)
        , dkfdc(ddkfdc) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;
    mdspan<gScalar, 1>       dkfdc;

    template <class T>
    CUDA_HOSTDEV void operator()(const T& i) const {
        return i.ddc(p, Temperature, c, dkfdc);
    }
};

struct ddTVisitor {

    CUDA_HOSTDEV
    ddTVisitor(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc)
        : p(pp)
        , Temperature(TT)
        , c(cc) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;

    template <class T>
    CUDA_HOSTDEV gScalar operator()(const T& i) const {
        return i.ddT(p, Temperature, c);
    }
};

struct everythingVisitor {

    CUDA_HOSTDEV
    everythingVisitor(gScalar                  pp,
                      gScalar                  TT,
                      mdspan<const gScalar, 1> cc,
                      mdspan<gScalar, 1>       ddkfdc)
        : p(pp)
        , Temperature(TT)
        , c(cc)
        , dkfdc(ddkfdc) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;
    mdspan<gScalar, 1>       dkfdc;

    template <class T>
    CUDA_HOSTDEV device_array<gScalar, 2>
                 operator()(const T& i) const {
        return i.everything(p, Temperature, c, dkfdc);
    }
};

struct gpuReactionRate {

    gpu::ReactionRate rate_;
    bool              irreversible_;

    gpuReactionRate() = default;

    gpuReactionRate(const gpu::ReactionRate& rate, bool irreversible)
        : rate_(rate)
        , irreversible_(irreversible) {}

    ///
    ///@brief Determines if the underlying reaction is irreversible or
    ///not.
    ///
    ///@return bool True if is irreversible, false otherwise.
    ///
    inline CUDA_HOSTDEV bool isIrreversible() const {
        return irreversible_;
    }

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
        using namespace gpu;

        RRVisitor v(p, T, c);

        return variant::apply_visitor(v, rate_);
    }

    ///
    ///@brief Determines if the rate is a function of concentration.
    ///
    ///@return True if is a function of concentration, false
    ///otherwise.
    ///
    inline CUDA_HOSTDEV bool hasDdc() const {

        return variant::apply_visitor(hasDdcVisitor(), rate_);
    }

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
                                 mdspan<gScalar, 1> dkfdc) const {
        ddcVisitor v(p, T, c, dkfdc);
        variant::apply_visitor(v, rate_);
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

        ddTVisitor v(p, T, c);
        return variant::apply_visitor(v, rate_);
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
    ///@param dkfdc A span of at least size nSpcecie where the
    /// derivative is put.
    ///@return device_array<gScalar, 2> A pair of scalars where the
    /// first value is the reaction rate and second value is the
    /// derivative w.r.t temperature.
    ///
    inline CUDA_HOSTDEV device_array<gScalar, 2>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             dkfdc) const {

        everythingVisitor v(p, T, c, dkfdc);
        return variant::apply_visitor(v, rate_);
    }
};

} // namespace FoamGpu