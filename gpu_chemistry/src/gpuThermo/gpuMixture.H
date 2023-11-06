#pragma once

#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuThermo.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

struct gpuMixture {

    const gpuThermo* thermos_;
    const gLabel     nSpecie_;

    gScalar rhoM_;
    gScalar CpM_;
    gScalar dCpMdT_;
    gScalar alphav_;

    CUDA_HOSTDEV gpuMixture(const gpuThermo* thermos,
                            const gLabel     nSpecie)
        : thermos_(thermos)
        , nSpecie_(nSpecie) {
        // computeAll();
    }

    CUDA_HOSTDEV const gpuThermo* specieThermos() const {
        return thermos_;
    }

    CUDA_HOSTDEV void computeAll(const mdspan<const gScalar, 1> y,
                                 const gScalar                  p,
                                 const gScalar                  T,
                                 mdspan<gScalar, 1>             Cp,
                                 mdspan<gScalar, 1>             Ha,
                                 mdspan<gScalar, 1>             c) {

        const gLabel nSpecie = y.size() - 2;

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
        rhoM_   = 1.0 / rhoM;
        CpM_    = CpM;
        dCpMdT_ = dCpMdT;
        alphav_ = alphav * rhoM_;

        for (gLabel i = 0; i < nSpecie; ++i) { c[i] *= rhoM_; }

        // alphav_ *= rhoM_;
    }

    CUDA_HOSTDEV gScalar rhoM() const { return rhoM_; }

    CUDA_HOSTDEV gScalar CpM() const { return CpM_; }

    CUDA_HOSTDEV gScalar dCpMdT() const { return dCpMdT_; }

    CUDA_HOSTDEV gScalar alphav() const { return alphav_; }
};

} // namespace FoamGpu
