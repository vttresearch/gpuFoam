#pragma once

#include "cuda_host_dev.H"
#include "double_intrinsics.H"
#include "gpuODESolverInputs.H"
#include "gpuODESystem.H"
#include "stepState.H"
#include <string>

namespace FoamGpu {

template <class System> class gpuODESolverBase {

protected:
    System system_;

    //- Absolute convergence tolerance per step
    gScalar absTol_;

    //- Relative convergence tolerance per step
    gScalar relTol_;

    //- The maximum number of sub-steps allowed for the integration step
    gLabel maxSteps_;

public:
    gpuODESolverBase() = default;

    /// @brief Construct from template ode system and solver inputs
    /// @param system the ODE system used
    /// @param i the tolerance inputs
    gpuODESolverBase(const System& system, const gpuODESolverInputs& i)
        : system_(system)
        , absTol_(i.absTol)
        , relTol_(i.relTol)
        , maxSteps_(i.maxSteps) {}

    /// @brief Get the absolute tolerance.
    /// @return Absolute tolerance.
    inline CUDA_HOSTDEV gScalar absTol() const { return absTol_; }

    /// @brief Get the relative tolerance.
    /// @return Relative tolerance.
    inline CUDA_HOSTDEV gScalar relTol() const { return relTol_; }

    /// @brief Get the maximum number of ode steps within single flow time step.
    /// @return The maximum number of steps for a single ode.
    inline CUDA_HOSTDEV gLabel maxSteps() const { return maxSteps_; }

    /// @brief Get the number of equations in the ode system.
    /// @return Number of equations in the ode system
    inline CUDA_HOSTDEV gLabel nEqns() const { return system_.nEqns(); }

    /// @brief Get a reference to the template ode system.
    /// @return A reference to the ode system.
    inline CUDA_HOSTDEV const System& system() const { return system_; }
};

/// @brief Iterate the input ode starting from time xStart to xEnd for vector y
/// using dxTry as an initial step estimate.
/// @param ode The ode to solve.
/// @param xStart Start time of the iteration.
/// @param xEnd End time of the iteration.
/// @param y The modified solution vector of size ode.nEqns().
/// @param dxTry Initial guess for the step size.
/// @param buffer A preallocated memory buffer.
template <class Ode>
static inline CUDA_HOSTDEV void odeSolve(const Ode&         ode,
                                         const gScalar      xStart,
                                         const gScalar      xEnd,
                                         mdspan<gScalar, 1> y,
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
        ode.solve(x, y, step, buffer);

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
