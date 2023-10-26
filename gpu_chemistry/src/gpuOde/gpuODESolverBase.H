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

} // namespace FoamGpu
