#pragma once

#include "check_ptr.H"
#include "cuda_host_dev.H"
#include "gpuBuffer.H"
#include "gpuMixture.H"
#include "gpuReaction.H"
#include "gpuThermo.H"
#include "gpu_constants.H"
#include "mdspan.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * //

namespace FoamGpu {

struct gpuODESystem {

private:
    gLabel             nEqns_;
    gLabel             nReactions_;
    const gpuThermo* __restrict__   thermos_;
    const gpuReaction* __restrict__ reactions_;

    /// @brief Get a pointer to the species thermos.
    /// @return A pointer to the specie thermos.
    CUDA_HOSTDEV const gpuThermo* specieThermos() const {
        return thermos_;
    }

    /// @brief Get a pointer to the reactions.
    /// @return A pointer to the reactions.
    CUDA_HOSTDEV const gpuReaction* reactions() const {
        return reactions_;
    }

    /// @brief Compute the derivative of y w.r.t. temperature.
    /// @param dydt The modified dY/dT vector of at least size
    /// nSpecie.
    /// @param rhoM Mixture density.
    CUDA_HOSTDEV void dYdT(mdspan<gScalar, 1> dydt,
                           const gScalar      rhoM) const {
        const gLabel nSpecie = nEqns() - 2;
        // Reactions return dNdtByV, so we need to convert the result
        // to dYdt
        for (gLabel i = 0; i < nSpecie; i++) {
            dydt[i] *= specieThermos()[i].W() / rhoM;
        }
    }

    // Temperature derivative, notes assumes dydx[0 ... nSpecie] has
    // been computed

    CUDA_HOSTDEV gScalar dTdt(const mdspan<const gScalar, 1> dydx,
                              const mdspan<const gScalar, 1> Ha,
                              const gScalar CpM) const {

        const gLabel nSpecie = nEqns() - 2;
        gScalar      ret     = 0.0;
        for (gLabel i = 0; i < nSpecie; i++) {
            ret -= dydx[i] * Ha[i];
        }
        return ret / CpM;
    }

    CUDA_HOSTDEV void ddY_dcdY(mdspan<gScalar, 2>             dfdy,
                               const mdspan<const gScalar, 1> dydx,
                               const mdspan<const gScalar, 1> c,
                               mdspan<gScalar, 1>             temp,
                               const gScalar                  rhoM,
                               const gScalar                  alphavM,
                               const gScalar                  p,
                               const gScalar T) const {
        const gLabel nSpecie = nEqns() - 2;

        for (gLabel i = 0; i < nSpecie; i++) {
            const gScalar WiByrhoM = specieThermos()[i].W() / rhoM;

            for (gLabel j = 0; j < (nSpecie + 2); ++j) {
                temp[j] = dfdy(i, j);
            }

            for (gLabel j = 0; j < nSpecie; j++) {
                const gScalar rhoMByWi =
                    rhoM / specieThermos()[j].W();

                const gScalar vj =
                    gScalar(1.0) / specieThermos()[j].rho(p, T);

                dfdy(i, j) = WiByrhoM * temp[j] * rhoMByWi +
                             rhoM * vj * dydx[i];
            }

            gScalar ddNidtByVdT = temp[nSpecie];

            for (gLabel j = 0; j < nSpecie; j++) {
                ddNidtByVdT -= temp[j] * c[j] * alphavM;
            }

            dfdy(i, nSpecie) =
                WiByrhoM * ddNidtByVdT + alphavM * dydx[i];
            dfdy(i, nSpecie + 1) = gScalar(0);
        }
    }

    // Assumes dfdy(0...nSpecie, 0...nSpecie) and dydx(0 ... nSpecie
    // +1 ) have been computed
    //  d(dTdt)/dY
    CUDA_HOSTDEV void ddT_dtdY(mdspan<gScalar, 2>             dfdy,
                               const mdspan<const gScalar, 1> dydx,
                               const mdspan<const gScalar, 1> Cp,
                               const mdspan<const gScalar, 1> Ha,
                               const gScalar CpM) const {

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
                               const gScalar dCpMdT) const {

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

    CUDA_HOSTDEV void derivatives(const mdspan<const gScalar, 1> y,
                                  mdspan<gScalar, 1>             dydx,
                                  gpuBuffer& buffer) const {

        for (gLabel i = 0; i < nEqns(); ++i) { dydx[i] = 0; }

        const gLabel nSpecie = nEqns() - 2;

        const gScalar T = y[nSpecie];
        const gScalar p = y[nSpecie + 1];

        gpuMixture mixture(thermos_);
        auto       Ha = buffer.tempField1();
        auto       c  = buffer.c();

        const auto pair = mixture.computeDerivativeRequired(y, Ha, c);
        const auto rhoM = pair[0];
        const auto CpM = pair[1]; 


        for (gLabel ri = 0; ri < nReactions(); ++ri) {
            reactions()[ri].dNdtByV(p, T, c, dydx);
        }

        // dY/dT
        dYdT(dydx, rhoM);

        // dT/dt
        dydx[nSpecie] = dTdt(dydx, Ha, CpM);

        // dp/dt
        dydx[nSpecie + 1] = gScalar(0);
    }

    CUDA_HOSTDEV void jacobian(const mdspan<const gScalar, 1> y,
                               mdspan<gScalar, 1>             dfdx,
                               mdspan<gScalar, 2>             dfdy,
                               gpuBuffer& buffer) const {

        for (int i = 0; i < nEqns(); ++i) { dfdx[i] = gScalar(0); }

        for (int i = 0; i < nEqns(); ++i) {
            for (int j = 0; j < nEqns(); ++j) {
                dfdy(i, j) = gScalar(0.0);
            }
        }

        const gLabel nSpecie = nEqns() - 2;

        const gScalar T = y[nSpecie];
        const gScalar p = y[nSpecie + 1];

        gpuMixture mixture(thermos_);
        auto       Cp = buffer.tempField1();
        auto       Ha = buffer.tempField2();
        auto       c  = buffer.c();

        const auto arr = mixture.computeJacobianRequired(y, Cp, Ha, c);
        const auto rhoM = arr[0];
        const auto CpM = arr[1];
        const auto dCpMdT = arr[2];
        const auto alphavM = arr[3];
            

        auto tempField1 = buffer.k4();

        auto ddc =
            mdspan<gScalar, 1>(&tempField1[0], extents<1>{nSpecie});



        for (gLabel ri = 0; ri < nReactions(); ++ri) {

            const auto& reaction = reactions()[ri];


            auto params =
                computeReactionParameters(reaction, c, p, T, ddc);

            reaction.dNdtByV(params, dfdx);

            reaction.ddNdtByVdcTp(p, T, c, dfdy, params);
        }

        // dY/dT
        dYdT(dfdx, rhoM);

        // dT/dt
        dfdx[nSpecie] = dTdt(dfdx, Ha, CpM);

        // dp/dt
        dfdx[nSpecie + 1] = gScalar(0);

        ////////////////////////////////

        // Temporary array to allow for re-using dfdy
        auto temp = buffer.k4();
        ddY_dcdY(dfdy, dfdx, c, temp, rhoM, alphavM, p, T);

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * //

} // namespace FoamGpu
