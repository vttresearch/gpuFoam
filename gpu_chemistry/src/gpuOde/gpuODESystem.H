#pragma once

#include "check_ptr.H"
#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "gpuBuffer.H"
#include "gpuReaction.H"
#include "gpuThermo.H"
#include "gpu_constants.H"
#include "mdspan.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu {

struct gpuODESystem {

private:
    gLabel             nEqns_;
    gLabel             nReactions_;
    const gpuThermo*   thermos_;
    const gpuReaction* reactions_;

    CUDA_HOSTDEV const gpuThermo* specieThermos() const { return thermos_; }

    CUDA_HOSTDEV const gpuReaction* reactions() const { return reactions_; }

    CUDA_HOSTDEV gScalar rhoMixture(const mdspan<const gScalar, 1> y,
                                    const gScalar                  p,
                                    const gScalar                  T) const {
        const gLabel nSpecie = nEqns() - 2;
        gScalar      rhoM    = 0.0;
        for (gLabel i = 0; i < nSpecie; i++) {
            rhoM += max(y[i], 0.0) / specieThermos()[i].rho(p, T);
        }
        return gScalar(1) / rhoM;
    }

    CUDA_HOSTDEV gScalar CpMixture(const mdspan<const gScalar, 1> y,
                                   const gScalar                  p,
                                   const gScalar                  T) const {
        const gLabel nSpecie = nEqns() - 2;
        gScalar      CpM     = 0.0;
        for (gLabel i = 0; i < nSpecie; i++) {
            CpM += max(y[i], 0.0) * specieThermos()[i].Cp(p, T);
        }
        return CpM;
    }

    CUDA_HOSTDEV gScalar dCpMdTMixture(const mdspan<const gScalar, 1> y,
                                       const gScalar                  p,
                                       const gScalar                  T) const {
        const gLabel nSpecie = nEqns() - 2;
        gScalar      dCpMdT  = 0.0;
        for (gLabel i = 0; i < nSpecie; i++) {
            dCpMdT += max(y[i], 0.0) * specieThermos()[i].dCpdT(p, T);
        }
        return dCpMdT;
    }

    // Mixture thermal expansion coefficient
    CUDA_HOSTDEV gScalar alphavMixture(const mdspan<const gScalar, 1> y,
                                       const gScalar                  p,
                                       const gScalar                  T,
                                       const gScalar rhoM) const {
        const gLabel nSpecie = nEqns() - 2;

        gScalar alphavM = gScalar(0);
        for (gLabel i = 0; i < nSpecie; i++) {
            const gScalar vi = gScalar(1.0) / specieThermos()[i].rho(p, T);
            const gScalar Yi = std::max(y[i], gScalar(0));
            alphavM += Yi * rhoM * vi * specieThermos()[i].alphav(p, T);
        }

        return alphavM;
    }

    CUDA_HOSTDEV void concentration(const mdspan<const gScalar, 1> y,
                                    mdspan<gScalar, 1>             c,
                                    const gScalar                  rhoM) const {
        const gLabel nSpecie = nEqns() - 2;
        // Evaluate the concentrations
        for (gLabel i = 0; i < nSpecie; i++) {
            c[i] = rhoM / specieThermos()[i].W() * max(y[i], 0.0);
        }
    }

    CUDA_HOSTDEV void absoluteEnthalpy(mdspan<gScalar, 1> Ha,
                                       const gScalar      p,
                                       const gScalar      T) const {
        const gLabel nSpecie = nEqns() - 2;
        for (gLabel i = 0; i < nSpecie; i++) {
            Ha[i] = specieThermos()[i].Ha(p, T);
        }
    }
    CUDA_HOSTDEV void heatCapacity(mdspan<gScalar, 1> Cp,
                                   const gScalar      p,
                                   const gScalar      T) const {
        const gLabel nSpecie = nEqns() - 2;
        for (gLabel i = 0; i < nSpecie; i++) {
            Cp[i] = specieThermos()[i].Cp(p, T);
        }
    }

    CUDA_HOSTDEV void dYdt(mdspan<gScalar, 1> dydx, const gScalar rhoM) const {
        const gLabel nSpecie = nEqns() - 2;
        // Reactions return dNdtByV, so we need to convert the result to dYdt
        for (gLabel i = 0; i < nSpecie; i++) {
            dydx[i] *= specieThermos()[i].W() / rhoM;
        }
    }

    // Temperature derivative, notes assumes dydx[0 ... nSpecie] has been
    // computed
    CUDA_HOSTDEV void dTdt(mdspan<gScalar, 1>             dydx,
                           const mdspan<const gScalar, 1> Ha,
                           const gScalar                  CpM) const {

        const gLabel nSpecie = nEqns() - 2;
        for (gLabel i = 0; i < nSpecie; i++) {
            dydx[nSpecie] -= dydx[i] * Ha[i];
        }
        dydx[nSpecie] /= CpM;
    }

    CUDA_HOSTDEV void dpdt(mdspan<gScalar, 1> dydx) const {
        const gLabel nSpecie = nEqns() - 2;
        // dp/dt = 0 (pressure is assumed constant)
        dydx[nSpecie + 1] = gScalar(0);
    }

    CUDA_HOSTDEV void ddY_dcdY(mdspan<gScalar, 2>             dfdy,
                               const mdspan<const gScalar, 1> dydx,
                               const mdspan<const gScalar, 1> c,
                               mdspan<gScalar, 1>             temp,
                               const gScalar                  rhoM,
                               const gScalar                  alphavM,
                               const gScalar                  p,
                               const gScalar                  T) const {
        const gLabel nSpecie = nEqns() - 2;

        for (gLabel i = 0; i < nSpecie; i++) {
            const gScalar WiByrhoM = specieThermos()[i].W() / rhoM;

            for (gLabel j = 0; j < (nSpecie + 2); ++j) { temp[j] = dfdy(i, j); }

            for (gLabel j = 0; j < nSpecie; j++) {
                const gScalar rhoMByWi = rhoM / specieThermos()[j].W();

                const gScalar vj = gScalar(1.0) / specieThermos()[j].rho(p, T);

                dfdy(i, j) =
                    WiByrhoM * temp[j] * rhoMByWi + rhoM * vj * dydx[i];
            }

            gScalar ddNidtByVdT = temp[nSpecie];

            for (gLabel j = 0; j < nSpecie; j++) {
                ddNidtByVdT -= temp[j] * c[j] * alphavM;
            }

            dfdy(i, nSpecie)     = WiByrhoM * ddNidtByVdT + alphavM * dydx[i];
            dfdy(i, nSpecie + 1) = gScalar(0);
        }
    }

    // Assumes dfdy(0...nSpecie, 0...nSpecie) and dydx(0 ... nSpecie +1 ) have
    // been computed
    //  d(dTdt)/dY
    CUDA_HOSTDEV void ddT_dtdY(mdspan<gScalar, 2>             dfdy,
                               const mdspan<const gScalar, 1> dydx,
                               const mdspan<const gScalar, 1> Cp,
                               const mdspan<const gScalar, 1> Ha,
                               const gScalar                  CpM) const {

        const gLabel nSpecie = nEqns() - 2;

        // d(dTdt)/dY
        for (gLabel i = 0; i < nSpecie; i++) {
            dfdy(nSpecie, i) = gScalar(0);
            for (gLabel j = 0; j < nSpecie; j++) {
                dfdy(nSpecie, i) -= dfdy(j, i) * Ha[j];
            }
            const auto Cpi = Cp[i];
            dfdy(nSpecie, i) -= Cpi * dydx[nSpecie];
            dfdy(nSpecie, i) /= CpM;
        }
    }

    // d(dTdt)/dT
    CUDA_HOSTDEV void ddT_dtdT(mdspan<gScalar, 2>             dfdy,
                               const mdspan<const gScalar, 1> dydx,
                               const mdspan<const gScalar, 1> Cp,
                               const mdspan<const gScalar, 1> Ha,
                               const gScalar                  CpM,
                               const gScalar                  dCpMdT) const {

        const gLabel nSpecie = nEqns() - 2;

        // d(dTdt)/dT
        dfdy(nSpecie, nSpecie) = gScalar(0);
        for (gLabel i = 0; i < nSpecie; i++) {
            dfdy(nSpecie, nSpecie) -=
                dydx[i] * Cp[i] + dfdy(i, nSpecie) * Ha[i];
        }
        dfdy(nSpecie, nSpecie) -= dydx[nSpecie] * dCpMdT;
        dfdy(nSpecie, nSpecie) /= CpM;
    }

public:
    gpuODESystem() = default;

    gpuODESystem(gLabel             nEqns,
                 gLabel             nReactions,
                 const gpuThermo*   thermos,
                 const gpuReaction* reactions)
        : nEqns_(nEqns)
        , nReactions_(nReactions)
        , thermos_(thermos)
        , reactions_(reactions) {}

    CUDA_HOSTDEV gLabel nEqns() const { return nEqns_; }

    CUDA_HOSTDEV gLabel nReactions() const { return nReactions_; }

    CUDA_HOSTDEV void derivatives(const gScalar                  x,
                                  const mdspan<const gScalar, 1> y,
                                  mdspan<gScalar, 1>             dydx,
                                  gpuBuffer&                     buffer) const {

        for (gLabel i = 0; i < nEqns(); ++i) { dydx[i] = 0; }

        const gLabel nSpecie = nEqns() - 2;

        const gScalar T    = y[nSpecie];
        const gScalar p    = y[nSpecie + 1];
        const gScalar rhoM = rhoMixture(y, p, T);
        const gScalar CpM  = CpMixture(y, p, T);

        auto c = buffer.c();
        concentration(y, c, rhoM);

        for (gLabel ri = 0; ri < nReactions(); ++ri) {
            reactions()[ri].dNdtByV(p, T, c, dydx);
        }

        // dY/dT
        dYdt(dydx, rhoM);

        // dT/dt
        auto Ha = buffer.tempField1();
        absoluteEnthalpy(Ha, p, T);
        dTdt(dydx, Ha, CpM);

        dpdt(dydx);
    }

    CUDA_HOSTDEV void jacobian(const gScalar                  x,
                               const mdspan<const gScalar, 1> y,
                               mdspan<gScalar, 1>             dfdx,
                               mdspan<gScalar, 2>             dfdy,
                               gpuBuffer&                     buffer) const {
        const gLabel nEqs    = nEqns();
        const gLabel nSpecie = nEqs - 2;

        auto tempField1 = buffer.tempField1();
        auto tempField2 = buffer.tempField2();

        const gScalar T = y[nSpecie];
        const gScalar p = y[nSpecie + 1];

        const gScalar rhoM   = rhoMixture(y, p, T);
        const gScalar CpM    = CpMixture(y, p, T);
        const gScalar dCpMdT = dCpMdTMixture(y, p, T);

        auto c = buffer.c();
        concentration(y, c, rhoM);

        // Evaluate the mixture thermal expansion coefficient
        const gScalar alphavM = alphavMixture(y, p, T, rhoM);

        // Evaluate contributions from reactions
        // dfdx = Zero;
        for (gLabel i = 0; i < gLabel(y.size()); ++i) { dfdx[i] = gScalar(0); }

        for (int i = 0; i < nEqs; ++i) {
            for (int j = 0; j < nEqs; ++j) { dfdy(i, j) = gScalar(0.0); }
        }

        for (gLabel ri = 0; ri < nReactions(); ++ri) {
            reactions()[ri].ddNdtByVdcTp(
                p, T, c, dfdx, dfdy, 0, nSpecie, tempField1, tempField2);
        }

        // dY/dT
        dYdt(dfdx, rhoM);

        dpdt(dfdx);

        // Temporary array to allow for re-using dfdy
        auto temp = buffer.tempField1();

        ddY_dcdY(dfdy, dfdx, c, temp, rhoM, alphavM, p, T);

        ////////////////////////////////
        // Here the derivative computation is finished because the temp arrays
        // are free
        //  and Cp/Ha can be cached
        //  dT/dt
        auto Cp = buffer.tempField2();
        heatCapacity(Cp, p, T);

        auto Ha = buffer.tempField1();
        absoluteEnthalpy(Ha, p, T);

        dTdt(dfdx, Ha, CpM);
        ////////////////////////////////

        // d(dTdt)/dY
        ddT_dtdY(dfdy, dfdx, Cp, Ha, CpM);

        // d(dTdt)/dT
        ddT_dtdT(dfdy, dfdx, Cp, Ha, CpM, dCpMdT);

        /*
        // d(dTdt)/dp = 0 (pressure is assumed constant)
        dfdy(nSpecie, nSpecie + 1) = gScalar(0);



        // d(dpdt)/dYiTp = 0 (pressure is assumed constant)
        for (gLabel i=0; i<nSpecie + 2; i++)
        {
            dfdy(nSpecie + 1, i) = gScalar(0);
        }
        */
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // namespace FoamGpu
