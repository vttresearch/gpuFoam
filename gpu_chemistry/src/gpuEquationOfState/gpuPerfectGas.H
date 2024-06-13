#pragma once

#include "cuda_host_dev.H"
#include "gpu_constants.H"
#include <cmath>

namespace FoamGpu {

class gpuPerfectGas {

public:
    //- Mass fraction of this specie in mixture
    gScalar Y_;

    //- Molecular weight of specie [kg/kmol]
    gScalar molWeight_;

    /// @brief Unary += operator. Increments the mass fraction
    /// directly and scales the molecular weight appropriately (?).
    /// @param st RHS gpuPerfectGas object
    inline void operator+=(const gpuPerfectGas& st) {

        const gScalar sumY = Y_ + st.Y_;
        if (std::abs(sumY) > gpuSmall) {
            molWeight_ =
                sumY / (Y_ / molWeight_ + st.Y_ / st.molWeight_);
        }

        Y_ = sumY;
    }

    /// @brief Unary *= operator. Scales the mass fraction with the
    /// input scalar.
    /// @param s The scalar used to scale the mass fraction.
    inline void operator*=(gScalar s) { Y_ *= s; }

public:
    // Constructors
    CUDA_HOSTDEV gpuPerfectGas()
        : Y_(0)
        , molWeight_(0) {}

    CUDA_HOSTDEV
    gpuPerfectGas(gScalar Y, gScalar molWeight)
        : Y_(Y)
        , molWeight_(molWeight) {}

    /// @brief Get the molecular weight of this specie [kg/kmol].
    /// @return The molecular weight of this specie.
    inline CUDA_HOSTDEV gScalar W() const { return molWeight_; }

    /// @brief Get the mass fraction of this specie in the the
    /// mixture.
    /// @return The mass fraction of this specie.
    inline CUDA_HOSTDEV gScalar Y() const { return Y_; }

    /// @brief Get the gas constant [J/kg/K].
    /// @return The gas constant of this specie.
    inline CUDA_HOSTDEV gScalar R() const {
        return gpuRR / molWeight_;
    }

    /// @brief Computes the density [kg/m^3] of this specie based on
    /// the input pressure and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The density of this specie.
    inline CUDA_HOSTDEV gScalar rho(gScalar p, gScalar T) const {
        return p / (this->R() * T);
    }

    /// @brief Computes the enthalpy [J/kg] contribution of this
    /// specie in the mixture based on the inpute pressure and
    /// temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The enthalpy contribution of this specie.
    inline CUDA_HOSTDEV gScalar H(const gScalar p,
                                  const gScalar T) const {
        (void)p;
        (void)T;
        return 0.0;
    }

    /// @brief Computes the heat capacity [J/(kg K)] contribution of
    /// this specie in the mixture in based on the input pressure and
    /// temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The heat capacity contribution of this specie.
    inline CUDA_HOSTDEV gScalar Cp(gScalar p, gScalar T) const {
        (void)p;
        (void)T;
        return 0.0;
    }

    /// @brief Computes the internal energy [J/kg] contribution  of
    /// this specie in the mixture based on the input pressure and
    /// temperature. Assumes contant pressure.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The heat internal energy contribution of this specie.
    inline CUDA_HOSTDEV gScalar E(const gScalar p,
                                  const gScalar T) const {
        (void)p;
        (void)T;
        return 0.0;
    }

    /// @brief Computes the heat capacity [J/(kg K)] contribution of
    /// this specie in the mixture based on the input pressure and
    /// temperature. Assumes constant volume.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The heat capacity contribution of this specie.
    inline CUDA_HOSTDEV gScalar Cv(gScalar p, gScalar T) const {
        (void)p;
        (void)T;
        return 0.0;
    }

    /// @brief Computes the entropy [J/kg/K] contribution to the
    /// integral of Cp/T in the mixture based on the input pressure
    /// and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The entropy contribution to the integral of Cp/T.
    inline CUDA_HOSTDEV gScalar Sp(const gScalar p,
                                   const gScalar T) const {
        (void)T;
        return -this->R() * std::log(p / gpuPstd);
    }

    /// @brief Computes the entropy [J/kg/K] contribution to the
    /// integral of Cv/T in the mixture based on the input pressure
    /// and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The entropy contribution to the integral of Cv/T.
    inline CUDA_HOSTDEV gScalar Sv(const gScalar p,
                                   const gScalar T) const {
        // throw std::logic_error("Not implemented");
        (void)p;
        (void)T;
        return 0.0;
    }

    /// @brief Computes the compressibility [s^2/m^2] of this specie
    /// in the mixture based on the input pressure and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The compressibility of this specie.
    inline CUDA_HOSTDEV gScalar psi(gScalar p, gScalar T) const {
        (void)p;
        return 1.0 / (this->R() * T);
    }

    /// @brief Computes the unitless compression factor of this specie
    /// based on the input pressure and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The compression factor of this specie.
    inline CUDA_HOSTDEV gScalar Z(gScalar p, gScalar T) const {
        (void)p;
        (void)T;
        return 1.0;
    }

    /// @brief Computes the difference (Cp - Cv) [J/(kg K)] based on
    /// the input pressure and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The difference Cp - Cv.
    inline CUDA_HOSTDEV gScalar CpMCv(gScalar p, gScalar T) const {
        (void)p;
        (void)T;
        return this->R();
    }

    /// @brief Computes the volumetric coefficient of thermal
    /// expansion [1/T] based on the input pressure and temperature.
    /// @param p Pressure.
    /// @param T Temperature.
    /// @return The volumetric coefficient of thhermal expansion.
    inline CUDA_HOSTDEV gScalar alphav(const gScalar p,
                                       const gScalar T) const {
        (void)p;
        return 1.0 / T;
    }
};

/// @brief Binary + operator for two gpuPerfectGas objects by summing
/// the mass fractions and molecular weights. Ensures that no zero
/// division occurs. Note! Host only.
/// @param st1 LHS gpuPerfectGas.
/// @param st2 RHS gpuPerfectGas.
/// @return A new gpuPerfectGas object with summed properties.
inline gpuPerfectGas operator+(const gpuPerfectGas& st1,
                               const gpuPerfectGas& st2) {
    const gScalar sumY = std::max(st1.Y_ + st2.Y_, gpuSmall);

    if (std::abs(sumY) > gpuSmall) {
        return gpuPerfectGas(sumY,
                             sumY / (st1.Y_ / st1.molWeight_ +
                                     st2.Y_ / st2.molWeight_));
    }

    return st1;
}

/// @brief Binary * operator for scalar and gpuPerfectGas object. The
/// multiplication is carried out only for the mass fraction. Note!
/// Host only.
/// @param s LHS scalar.
/// @param st RHS gpuPerfectGas.
/// @return A new gpuPerfectGas object where the mass fraction has
/// been scaled by the input scalar.
inline gpuPerfectGas operator*(gScalar s, const gpuPerfectGas& st) {
    return gpuPerfectGas(s * st.Y_, st.molWeight_);
}

/// @brief Binary == operator for two gpuPerfectGas objects. The mass
/// fraction and molecular weight of the retuned new object has the
/// difference of the mass fractions and molecular weights.
/// @param st1 LHS gpuPerfectGas.
/// @param st2 RHS gpuPerfectGas.
/// @return A new gpuPerfectGas object with (st2.Y - st1.Y) as a mass
/// fraction and scaled difference of the molecular weights as the
/// molecular weight.
inline gpuPerfectGas operator==(const gpuPerfectGas& st1,
                                const gpuPerfectGas& st2) {

    gScalar diffY = st2.Y_ - st1.Y_;
    if (std::abs(diffY) < gpuSmall) { diffY = gpuSmall; }

    const gScalar diffRW =
        st2.Y_ / st2.molWeight_ - st1.Y_ / st1.molWeight_;

    gScalar molWeight = gpuGreat;
    if (std::abs(diffRW) > gpuSmall) { molWeight = diffY / diffRW; }

    return gpuPerfectGas(diffY, molWeight);
}

} // namespace FoamGpu
