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

struct my_visitor1 {

    CUDA_HOSTDEV
    my_visitor1(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc)
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

struct my_visitor2 {

    my_visitor2() = default;

    template <class T>
    CUDA_HOSTDEV bool operator()(const T& i) const {
        return i.hasDdc();
    }
};

struct my_visitor3 {

    CUDA_HOSTDEV
    my_visitor3(gScalar                  pp,
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

struct my_visitor4 {

    CUDA_HOSTDEV
    my_visitor4(gScalar pp, gScalar TT, mdspan<const gScalar, 1> cc)
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

struct my_visitor5 {

    CUDA_HOSTDEV
    my_visitor5(gScalar                  pp,
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
    CUDA_HOSTDEV std::pair<gScalar, gScalar> operator()(const T& i) const {
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

    inline CUDA_HOSTDEV bool isIrreversible() const {
        return irreversible_;
    }

    inline CUDA_HOSTDEV gScalar
    operator()(const gScalar                  p,
               const gScalar                  T,
               const mdspan<const gScalar, 1> c) const {
        using namespace gpu;

        my_visitor1 v(p, T, c);

        // auto v = [=] CUDA_HOSTDEV (const auto& i){return
        // i.operator()(p, T, c, li);};

        return variant::apply_visitor(v, rate_);
    }

    inline CUDA_HOSTDEV bool hasDdc() const {
        /*
        auto visitor = [] CUDA_HOSTDEV (const auto& i){
            return i.hasDdc();
        };
        return variant::apply_visitor(visitor, rate_);
        */
        return variant::apply_visitor(my_visitor2(), rate_);
        // return
        // std::visit(my_visitor2(), rate_);
    }

    inline CUDA_HOSTDEV void ddc(const gScalar                  p,
                                 const gScalar                  T,
                                 const mdspan<const gScalar, 1> c,
                                 mdspan<gScalar, 1> dkfdc) const {
        my_visitor3 v(p, T, c, dkfdc);
        variant::apply_visitor(v, rate_);
        // std::visit(v, rate_);
    }

    inline CUDA_HOSTDEV gScalar
    ddT(const gScalar                  p,
        const gScalar                  T,
        const mdspan<const gScalar, 1> c) const {

        my_visitor4 v(p, T, c);
        return variant::apply_visitor(v, rate_);
        // return std::visit(v, rate_);
        // return
        // std::visit([&](auto&& arg){return arg.ddT(p, T, c, li);},
        // rate_);
    }

    inline CUDA_HOSTDEV std::pair<gScalar, gScalar>
                        everything(const gScalar                  p,
                                   const gScalar                  T,
                                   const mdspan<const gScalar, 1> c,
                                   mdspan<gScalar, 1>             dkfdc) const {
        /*
        auto k   = this->operator()(p, T, c);
        auto ddT = this->ddT(p, T, c);
        if (this->hasDdc()) { this->ddc(p, T, c, dkfdc); }
        return std::make_pair(k, ddT);
        */
        my_visitor5 v(p, T, c, dkfdc);
        return variant::apply_visitor(v, rate_);

    }
};

} // namespace FoamGpu