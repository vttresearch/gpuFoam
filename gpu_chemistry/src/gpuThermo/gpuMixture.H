#pragma once

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuThermo.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

struct gpuMixture {

    const gpuThermo* thermos_;

    CUDA_HOSTDEV gpuMixture(const gpuThermo* thermos)
        : thermos_(thermos) {}

    CUDA_HOSTDEV const gpuThermo* specieThermos() const {
        return thermos_;
    }

    /// @brief Computes the mixture properties required by the
    /// Jacobian calculation.
    /// @param y [IN] A span containing specie mass fractions,
    /// temperature and pressure.
    /// @param Cp [OUT] A span containing the species heat capacities.
    /// @param Ha [OUT] A span containing the species absolute
    /// enthalpies.
    /// @param c [OUT] A span containing the species concetrations.
    /// @return std::tuple<gScalar, gScalar, gScalar, gScalar> a tuple
    /// tuple of mixture density, heat capacity, derivative of heat
    /// capacity w.r.t temperature and volumetric coefficient of
    /// thermal expansion, respectively.
    CUDA_HOSTDEV std::tuple<gScalar, gScalar, gScalar, gScalar>
                 computeJacobianRequired(const mdspan<const gScalar, 1> y,
                                         mdspan<gScalar, 1>             Cp,
                                         mdspan<gScalar, 1>             Ha,
                                         mdspan<gScalar, 1>             c) {

        const gLabel  nSpecie = y.size() - 2;
        const gScalar T       = y[nSpecie];
        const gScalar p       = y[nSpecie + 1];

        gScalar rhoM   = 0.0;
        gScalar CpM    = 0.0;
        gScalar dCpMdT = 0.0;
        gScalar alphav = 0.0;

        for (gLabel i = 0; i < nSpecie; ++i) {
            const gScalar    yi     = max(y[i], 0.0);
            const gpuThermo& thermo = specieThermos()[i];
            const gScalar    rhoi   = thermo.rho(p, T);
            const gScalar    vi     = 1.0 / rhoi;

            const gScalar* coeffs  = thermo.coeffs(T);
            const gScalar  Cpi     = thermo.Cp(p, T, coeffs);
            const gScalar  Hai     = thermo.Ha(p, T, coeffs);
            const gScalar  dCpMdTi = thermo.dCpdT(p, T, coeffs);

            rhoM += yi / rhoi;
            CpM += yi * Cpi;
            dCpMdT += yi * dCpMdTi;
            alphav += yi * vi * thermo.alphav(p, T);
            Cp[i] = Cpi;
            Ha[i] = Hai;
            c[i]  = yi / thermo.W();
        }
        rhoM   = 1.0 / rhoM;
        CpM    = CpM;
        dCpMdT = dCpMdT;
        alphav = alphav * rhoM;

        for (gLabel i = 0; i < nSpecie; ++i) { c[i] *= rhoM; }

        return std::make_tuple(rhoM, CpM, dCpMdT, alphav);

        // alphav_ *= rhoM_;
    }

    /// @brief Computes the mixture properties required by the
    /// derivative calculation.
    /// @param y [IN] A span containing specie mass fractions,
    /// temperature and pressure.
    /// @param Ha [OUT] A span containing the species absolute
    /// enthalpies.
    /// @param c [OUT] A span containing the species concetrations.
    /// @return std::pair<gScalar, gScalar> a pair containing mixture
    /// density and specific heat, respectively.
    CUDA_HOSTDEV std::pair<gScalar, gScalar>
                 computeDerivativeRequired(const mdspan<const gScalar, 1> y,
                                           mdspan<gScalar, 1>             Ha,
                                           mdspan<gScalar, 1>             c) {

        const gLabel  nSpecie = y.size() - 2;
        const gScalar T       = y[nSpecie];
        const gScalar p       = y[nSpecie + 1];

        gScalar rhoM = 0.0;
        gScalar CpM  = 0.0;

        for (gLabel i = 0; i < nSpecie; ++i) {
            const gScalar    yi     = max(y[i], 0.0);
            const gpuThermo& thermo = specieThermos()[i];
            const gScalar    rhoi   = thermo.rho(p, T);

            const gScalar* coeffs = thermo.coeffs(T);
            const gScalar  Cpi    = thermo.Cp(p, T, coeffs);
            const gScalar  Hai    = thermo.Ha(p, T, coeffs);

            rhoM += yi / rhoi;
            CpM += yi * Cpi;
            Ha[i] = Hai;
            c[i]  = yi / thermo.W();
        }
        rhoM = 1.0 / rhoM;

        for (gLabel i = 0; i < nSpecie; ++i) { c[i] *= rhoM; }

        return std::make_pair(rhoM, CpM);

        // alphav_ *= rhoM_;
    }
};

} // namespace FoamGpu
