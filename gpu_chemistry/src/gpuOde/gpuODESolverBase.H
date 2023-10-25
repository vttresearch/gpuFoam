#pragma once


#include "cuda_host_dev.H"
#include "cuda_math.H"
#include "cuda_math.H"
#include "double_intrinsics.H"
#include "gpuODESystem.H"
#include "gpuOdeSolve.H"
#include "stepState.H"
#include <string>

namespace FoamGpu {




template <class Derived> class gpuODESolverBase {

protected:

    //- Absolute convergence tolerance per step
    gScalar absTol_;

    //- Relative convergence tolerance per step
    gScalar relTol_;

    //- The maximum number of sub-steps allowed for the integration step
    gLabel maxSteps_;

private:
    // CRTP interface for Derived::solve
    CUDA_HOSTDEV void interface(gScalar&            x,
                                mdspan<gScalar, 1> y,
                                const gLabel        li,
                                stepState&          step,
                                gpuBuffer&          buffer) const {

        static_cast<const Derived*>(this)->solve(x, y, li, step, buffer);
    }

public:
    // Constructors
    gpuODESolverBase() = default;

    //- Construct for given ODESystem specifying tolerances
    gpuODESolverBase(const gpuODESolverInputs& i)
        : absTol_(i.absTol)
        , relTol_(i.relTol)
        , maxSteps_(i.maxSteps) {}

    //- Solve the ODE system from the current state xStart, y
    //  and the optional index into the list of systems to solve li
    //  to xEnd and return an estimate for the next step in dxTry

    CUDA_HOSTDEV void solve(const gScalar       xStart,
                            const gScalar       xEnd,
                            mdspan<gScalar, 1> y,
                            const gLabel        li,
                            gScalar&            dxTry,
                            gpuBuffer&          buffer) const {

        //odeSolve(*this, xStart, xEnd, y, li, dxTry, buffer, maxSteps_);

        stepState step(dxTry);
        gScalar   x = xStart;

        // printf("dxTry value = %lf\n", dxTry);

        for (gLabel nStep = 0; nStep < maxSteps_; nStep++) {

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
            interface(x, y, li, step, buffer);

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

    //- Return access to the absolute tolerance field
    inline CUDA_HOSTDEV gScalar absTol() const { return absTol_; }

    //- Return access to the relative tolerance field
    inline CUDA_HOSTDEV gScalar relTol() const { return relTol_; }
};

} // namespace FoamGpu
