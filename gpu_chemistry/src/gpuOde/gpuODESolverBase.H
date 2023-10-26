#pragma once


#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "cuda_math.H"
#include "double_intrinsics.H"
#include "gpuODESystem.H"
#include "stepState.H"
#include <string>

namespace FoamGpu {




class gpuODESolverBase {

protected:

    //- Absolute convergence tolerance per step
    gScalar absTol_;

    //- Relative convergence tolerance per step
    gScalar relTol_;

    //- The maximum number of sub-steps allowed for the integration step
    gLabel maxSteps_;



public:
    // Constructors
    gpuODESolverBase() = default;

    //- Construct for given ODESystem specifying tolerances
    gpuODESolverBase(const gpuODESolverInputs& i)
        : absTol_(i.absTol)
        , relTol_(i.relTol)
        , maxSteps_(i.maxSteps) {}


    inline CUDA_HOSTDEV gScalar absTol() const { return absTol_; }

    inline CUDA_HOSTDEV gScalar relTol() const { return relTol_; }

    inline CUDA_HOSTDEV gLabel maxSteps() const {return maxSteps_;}

};

// This is the main solve call from gpuODESolverBase
template <class Ode>
static inline CUDA_HOSTDEV void odeSolve(const Ode&         ode,
                                         const gScalar      xStart,
                                         const gScalar      xEnd,
                                         mdspan<gScalar, 1> y,
                                         const gLabel       li,
                                         gScalar&           dxTry,
                                         gpuBuffer&         buffer) {

    stepState step(dxTry);
    gScalar   x = xStart;

    // printf("dxTry value = %lf\n", dxTry);

    for (gLabel nStep = 0; nStep < ode.maxSteps(); nStep++) {

        // Store previous iteration dxTry
        gScalar dxTry0 = step.dxTry;

        step.reject = false;

        // Check if this is a truncated step and set dxTry to integrate to
        // xEnd
        if ((x + step.dxTry - xEnd) * (x + step.dxTry - xStart) > 0) {
            step.last  = true;
            step.dxTry = xEnd - x;
        }

        // Integrate as far as possible up to step.dxTry
        ode.solve(x, y, li, step, buffer);

        // Check if reached xEnd
        if ((x - xEnd) * (xEnd - xStart) >= 0) {
            if (nStep > 0 && step.last) { step.dxTry = dxTry0; }

            dxTry = step.dxTry;

            return;
        }

        step.first = false;

        // If the step.dxTry was reject set step.prevReject
        if (step.reject) { step.prevReject = true; }
    }
}



} // namespace FoamGpu
