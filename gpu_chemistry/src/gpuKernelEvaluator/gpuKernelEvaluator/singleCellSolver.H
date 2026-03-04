#pragma once

#include "gpuBuffer.H"
#include "gpuODESolver.H"
#include "gpu_constants.H"
#include "mdspan.H"

namespace FoamGpu {

struct singleCellSolver {

    singleCellSolver(gScalar              deltaT,
                     gLabel               nSpecie,
                     mdspan<gScalar, 1>   deltaTChem,
                     mdspan<gScalar, 2>   Yvf,
                     mdspan<gpuBuffer, 1> buffer,
                     gpuODESolver         ode,
                     gScalar              Treact = 0.0)
        : deltaT_(deltaT)
        , nSpecie_(nSpecie)
        , deltaTChem_(deltaTChem)
        , Yvf_(Yvf)
        , buffer_(buffer)
        , ode_(ode)
        , Treact_(Treact) {}

    CUDA_HOSTDEV void operator()(gLabel celli) const {
        auto Y = mdspan<gScalar, 1>(&Yvf_(celli, 0),
                                    extents<1>{nSpecie_ + 2});

        if (Y[nSpecie_] > Treact_) {
            // Initialise time progress
            gScalar timeLeft = deltaT_;

            // Calculate the chemical source terms
            while (timeLeft > gpuSmall) {
                gScalar dt = timeLeft;

                ode_.solve(0, dt, Y, deltaTChem_[celli], buffer_[celli]);

                for (int i = 0; i < nSpecie_; i++) {
                    Y[i] = fmax(0.0, Y[i]);
                }

                timeLeft -= dt;
            }
        }
        else {
            deltaTChem_[celli] = deltaT_; // set the chemistry time step to the flow time step
        }
    }

    gScalar              deltaT_;
    gLabel               nSpecie_;
    mdspan<gScalar, 1>   deltaTChem_;
    mdspan<gScalar, 2>   Yvf_;
    mdspan<gpuBuffer, 1> buffer_;
    gpuODESolver         ode_;
    gScalar              Treact_;
};

} // namespace FoamGpu