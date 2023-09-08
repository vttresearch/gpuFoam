#pragma once

#include "cuda_host_dev.H"
#include "gpuConstants.H"
#include "gpuPerfectGas.H"
#include <array> //std::array

namespace FoamGpu {

class gpuThermo : public gpuPerfectGas {
public:
    using EquationOfState = gpuPerfectGas;

    static constexpr size_t nCoeffs_ = 7;

    using coeffArray = std::array<gScalar, nCoeffs_>;

    // Temperature limits of applicability of functions
    gScalar Tlow_, Thigh_, Tcommon_;

    coeffArray highCpCoeffs_;
    coeffArray lowCpCoeffs_;

    inline CUDA_HOSTDEV const gScalar* coeffs(gScalar T) const {
        if (T < Tcommon_) { return lowCpCoeffs_.data(); }
        return highCpCoeffs_.data();
    }

public:
    // Constructors

    //- Construct null
    gpuThermo() = default;

    gpuThermo(const EquationOfState& eos,
              gScalar                Tlow,
              gScalar                Thigh,
              gScalar                Tcommon,
              const coeffArray&      highCpCoeffs,
              const coeffArray&      lowCpCoeffs)
        : EquationOfState(eos)
        , Tlow_(Tlow)
        , Thigh_(Thigh)
        , Tcommon_(Tcommon)
        , highCpCoeffs_(highCpCoeffs)
        , lowCpCoeffs_(lowCpCoeffs) {}

    gpuThermo(gScalar    Y,
              gScalar    molWeight,
              gScalar    Tlow,
              gScalar    Thigh,
              gScalar    Tcommon,
              coeffArray highCoeffs,
              coeffArray lowCoeffs)
        : EquationOfState(Y, molWeight)
        , Tlow_(Tlow)
        , Thigh_(Thigh)
        , Tcommon_(Tcommon)
        , highCpCoeffs_(highCoeffs)
        , lowCpCoeffs_(lowCoeffs) {}

    inline CUDA_HOSTDEV const coeffArray& highCpCoeffs() const {
        return highCpCoeffs_;
    }
    inline CUDA_HOSTDEV const coeffArray& lowCpCoeffs() const {
        return lowCpCoeffs_;
    }

    //- Equilibrium constant [] i.t.o fugacities
    //  = PIi(fi/Pstd)^nui
    inline CUDA_HOSTDEV gScalar K(const gScalar p, const gScalar T) const {

        (void)p;

        gScalar arg = -this->Y() * this->Gstd(T) / (gpuRR * T);

        if (arg < 600) { return std::exp(arg); }
        return std::sqrt(gpuVGreat);
        // return gpuRootVGreat;
    }

    //- Equilibrium constant [] i.t.o. partial pressures
    //  = PIi(pi/Pstd)^nui
    //  For low pressures (where the gas mixture is near perfect) Kp = K
    inline CUDA_HOSTDEV gScalar Kp(const gScalar p, const gScalar T) const {
        return K(p, T);
    }

    //- Equilibrium constant i.t.o. molar concentration
    //  = PIi(ci/cstd)^nui
    //  For low pressures (where the gas mixture is near perfect)
    //  Kc = Kp(pstd/(RR*T))^nu
    inline CUDA_HOSTDEV gScalar Kc(const gScalar p, const gScalar T) const {

        const gScalar nm = this->Y() / this->W();

        if (std::abs(nm - gpuSmall) < gpuVSmall) { return Kp(p, T); }
        return Kp(p, T) * std::pow(gpuPstd / (gpuRR * T), nm);
    }

    //- Derivative of B (according to Niemeyer et al.)
    //  w.r.t. temperature
    inline CUDA_HOSTDEV gScalar dKcdTbyKc(const gScalar p,
                                          const gScalar T) const {

        const gScalar dKcdTbyKc =
            (this->S(gpuPstd, T) + this->Gstd(T) / T) * this->Y() / (gpuRR * T);

        const gScalar nm = this->Y() / this->W();

        if (std::abs(nm - gpuSmall) < 1E-8) { return dKcdTbyKc; }
        return dKcdTbyKc - nm / T;
    }

    //- Heat capacity at constant pressure [J/kg/K]
    inline CUDA_HOSTDEV gScalar Cp(const gScalar p, const gScalar T) const {

        const auto& a = coeffs(T);
        return ((((a[4] * T + a[3]) * T + a[2]) * T + a[1]) * T + a[0]) +
               EquationOfState::Cp(p, T);
    }

    //- Absolute enthalpy [J/kg]
    inline CUDA_HOSTDEV gScalar Ha(const gScalar p, const gScalar T) const {
        const auto& a = coeffs(T);
        return (((((a[4] / 5.0 * T + a[3] / 4.0) * T + a[2] / 3.0) * T +
                  a[1] / 2.0) *
                     T +
                 a[0]) *
                    T +
                a[5]) +
               EquationOfState::H(p, T);
    }

    //- Sensible enthalpy [J/kg]
    inline CUDA_HOSTDEV gScalar Hs(const gScalar p, const gScalar T) const {
        return Ha(p, T) - Hf();
    }

    //- Enthalpy of formation [J/kg]
    inline CUDA_HOSTDEV gScalar Hf() const {

        const auto& a = lowCpCoeffs_;
        return (((((a[4] / 5.0 * gpuTstd + a[3] / 4.0) * gpuTstd + a[2] / 3.0) *
                      gpuTstd +
                  a[1] / 2.0) *
                     gpuTstd +
                 a[0]) *
                    gpuTstd +
                a[5]);
    }

    //- Entropy [J/kg/K]
    inline CUDA_HOSTDEV gScalar S(const gScalar p, const gScalar T) const {
        const auto& a = coeffs(T);
        return ((((a[4] / 4.0 * T + a[3] / 3.0) * T + a[2] / 2.0) * T + a[1]) *
                    T +
                a[0] * std::log(T) + a[6]) +
               EquationOfState::Sp(p, T);
    }

    //- Gibbs free energy of the mixture in the standard state [J/kg]
    inline CUDA_HOSTDEV gScalar Gstd(const gScalar T) const {
        const auto& a = coeffs(T);

        return ((a[0] * (1 - std::log(T)) -
                 (((a[4] / 20.0 * T + a[3] / 12.0) * T + a[2] / 6.0) * T +
                  a[1] / 2.0) *
                     T -
                 a[6]) *
                    T +
                a[5]);
    }

    inline CUDA_HOSTDEV gScalar Cv(const gScalar p, const gScalar T) const {
        return Cp(p, T) - EquationOfState::CpMCv(p, T);
    }

    inline CUDA_HOSTDEV gScalar Es(const gScalar p, const gScalar T) const {
        return Hs(p, T) - p / EquationOfState::rho(p, T);
    }

    inline CUDA_HOSTDEV gScalar Ea(const gScalar p, const gScalar T) const {
        return Ha(p, T) - p / EquationOfState::rho(p, T);
    }

    //- Temperature derivative of heat capacity at constant pressure
    inline CUDA_HOSTDEV gScalar dCpdT(const gScalar p, const gScalar T) const {
        const auto& a = coeffs(T);
        return (((4 * a[4] * T + 3 * a[3]) * T + 2 * a[2]) * T + a[1]);
    }

    inline void operator+=(const gpuThermo& jt) {

        gScalar Y1 = this->Y();

        EquationOfState::operator+=(jt);

        if (std::abs(this->Y()) > gpuSmall) {
            Y1 /= this->Y();
            const gScalar Y2 = jt.Y() / this->Y();

            Tlow_  = std::max(Tlow_, jt.Tlow_);
            Thigh_ = std::min(Thigh_, jt.Thigh_);

            for (size_t coefLabel = 0; coefLabel < nCoeffs_; coefLabel++) {
                highCpCoeffs_[coefLabel] = Y1 * highCpCoeffs_[coefLabel] +
                                           Y2 * jt.highCpCoeffs_[coefLabel];

                lowCpCoeffs_[coefLabel] = Y1 * lowCpCoeffs_[coefLabel] +
                                          Y2 * jt.lowCpCoeffs_[coefLabel];
            }
        }
    }
};

inline gpuThermo operator+(const gpuThermo& jt1, const gpuThermo& jt2) {

    gpuThermo::EquationOfState eofs = jt1;
    eofs += jt2;

    if (std::abs(eofs.Y_) < gpuSmall) {
        return gpuThermo(eofs,
                         jt1.Tlow_,
                         jt1.Thigh_,
                         jt1.Tcommon_,
                         jt1.highCpCoeffs_,
                         jt1.lowCpCoeffs_);
    }

    const gScalar Y1 = jt1.Y() / eofs.Y();
    const gScalar Y2 = jt2.Y() / eofs.Y();

    typename gpuThermo::coeffArray highCpCoeffs{};
    typename gpuThermo::coeffArray lowCpCoeffs{};

    for (size_t coefLabel = 0; coefLabel < gpuThermo::nCoeffs_; coefLabel++) {
        highCpCoeffs[coefLabel] = Y1 * jt1.highCpCoeffs_[coefLabel] +
                                  Y2 * jt2.highCpCoeffs_[coefLabel];

        lowCpCoeffs[coefLabel] =
            Y1 * jt1.lowCpCoeffs_[coefLabel] + Y2 * jt2.lowCpCoeffs_[coefLabel];
    }

    return gpuThermo(eofs,
                     std::max(jt1.Tlow_, jt2.Tlow_),
                     std::min(jt1.Thigh_, jt2.Thigh_),
                     jt1.Tcommon_,
                     highCpCoeffs,
                     lowCpCoeffs);
}

inline gpuThermo operator*(gScalar s, const gpuThermo& jt) {
    return gpuThermo(s * static_cast<const gpuThermo::EquationOfState&>(jt),
                     jt.Tlow_,
                     jt.Thigh_,
                     jt.Tcommon_,
                     jt.highCpCoeffs_,
                     jt.lowCpCoeffs_);
}

inline gpuThermo operator==(const gpuThermo& jt1, const gpuThermo& jt2) {

    gpuThermo::EquationOfState eofs(
        static_cast<const gpuThermo::EquationOfState&>(jt1) ==
        static_cast<const gpuThermo::EquationOfState&>(jt2));

    const gScalar Y1 = jt2.Y() / eofs.Y();
    const gScalar Y2 = jt1.Y() / eofs.Y();

    typename gpuThermo::coeffArray highCpCoeffs{};
    typename gpuThermo::coeffArray lowCpCoeffs{};

    for (size_t coefLabel = 0; coefLabel < gpuThermo::nCoeffs_; coefLabel++) {
        highCpCoeffs[coefLabel] = Y1 * jt2.highCpCoeffs_[coefLabel] -
                                  Y2 * jt1.highCpCoeffs_[coefLabel];

        lowCpCoeffs[coefLabel] =
            Y1 * jt2.lowCpCoeffs_[coefLabel] - Y2 * jt1.lowCpCoeffs_[coefLabel];
    }

    return gpuThermo(eofs,
                     std::max(jt2.Tlow_, jt1.Tlow_),
                     std::min(jt2.Thigh_, jt1.Thigh_),
                     jt2.Tcommon_,
                     highCpCoeffs,
                     lowCpCoeffs);
}

} // namespace FoamGpu
