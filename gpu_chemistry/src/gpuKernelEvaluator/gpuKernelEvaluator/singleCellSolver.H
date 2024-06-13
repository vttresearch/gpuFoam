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
                     gpuODESolver         ode)
        : deltaT_(deltaT)
        , nSpecie_(nSpecie)
        , deltaTChem_(deltaTChem)
        , Yvf_(Yvf)
        , buffer_(buffer)
        , ode_(ode) {}

    CUDA_HOSTDEV void operator()(gLabel celli) const {
        auto Y = mdspan<gScalar, 1>(&Yvf_(celli, 0),
                                    extents<1>{nSpecie_ + 2});

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

    gScalar              deltaT_;
    gLabel               nSpecie_;
    mdspan<gScalar, 1>   deltaTChem_;
    mdspan<gScalar, 2>   Yvf_;
    mdspan<gpuBuffer, 1> buffer_;
    gpuODESolver         ode_;
};

} // namespace FoamGpu