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
    gpuFallOffReactionRate<gpuArrheniusReactionRate, gpuTroeFallOffFunction>;

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

struct my_visitor1 {

    CUDA_HOSTDEV
    my_visitor1(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc, gLabel lli)
        : p(pp)
        , Temperature(TT)
        , c(cc)
        , li(lli) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;
    gLabel                   li;

    template <class T> CUDA_HOSTDEV gScalar operator()(const T& i) const {
        return i.operator()(p, Temperature, c, li);
    }
};

struct my_visitor2 {

    my_visitor2() = default;

    template <class T> CUDA_HOSTDEV bool operator()(const T& i) const {
        return i.hasDdc();
    }
};

struct my_visitor3 {

    CUDA_HOSTDEV
    my_visitor3(gScalar                  pp,
                gScalar                  TT,
                mdspan<const gScalar, 1> cc,
                gLabel                   lli,
                mdspan<gScalar, 1>       ddkfdc)
        : p(pp)
        , Temperature(TT)
        , c(cc)
        , li(lli)
        , dkfdc(ddkfdc) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;
    gLabel                   li;
    mdspan<gScalar, 1>       dkfdc;

    template <class T> CUDA_HOSTDEV void operator()(const T& i) const {
        return i.ddc(p, Temperature, c, li, dkfdc);
    }
};

struct my_visitor4 {

    CUDA_HOSTDEV
    my_visitor4(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc, gLabel lli)
        : p(pp)
        , Temperature(TT)
        , c(cc)
        , li(lli) {}

    gScalar                  p;
    gScalar                  Temperature;
    mdspan<const gScalar, 1> c;
    gLabel                   li;

    template <class T> CUDA_HOSTDEV gScalar operator()(const T& i) const {
        return i.ddT(p, Temperature, c, li);
    }
};

struct gpuReactionRate {

    gpu::ReactionRate rate_;
    bool              irreversible_;

    gpuReactionRate() = default;

    gpuReactionRate(const gpu::ReactionRate& rate, bool irreversible)
        : rate_(rate)
        , irreversible_(irreversible) {}

    inline CUDA_HOSTDEV bool isIrreversible() const { return irreversible_; }

    inline CUDA_HOSTDEV gScalar operator()(const gScalar                  p,
                                           const gScalar                  T,
                                           const mdspan<const gScalar, 1> c,
                                           const gLabel li) const {
        using namespace gpu;

        my_visitor1 v(p, T, c, li);
        // return std::visit(v, rate_);
        return variant::apply_visitor(v, rate_);
    }

    inline CUDA_HOSTDEV bool hasDdc() const {
        return variant::apply_visitor(my_visitor2(), rate_);
        // return
        // std::visit(my_visitor2(), rate_);
    }

    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 const gLabel                   li,
                                 mdspan<gScalar, 1>             dkfdc) const {
        my_visitor3 v(p, T, c, li, dkfdc);
        variant::apply_visitor(v, rate_);
        // std::visit(v, rate_);
    }

    inline CUDA_HOSTDEV gScalar ddT(const gScalar                  p,
                                    const gScalar                  T,
                                    const mdspan<const gScalar, 1> c,
                                    const gLabel                   li) const {

        my_visitor4 v(p, T, c, li);
        return variant::apply_visitor(v, rate_);
        // return std::visit(v, rate_);
        // return
        // std::visit([&](auto&& arg){return arg.ddT(p, T, c, li);}, rate_);
    }
};

} // namespace FoamGpu